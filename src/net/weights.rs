//! The parameters (`model.safetensors`), read into typed fp32 tensors.
//!
//! Names are the PyTorch `state_dict` names, shapes are checked against the
//! [`Architecture`], and every tensor the forward pass needs must be present. Backends take
//! these fp32 arrays and convert to their own layout and precision at load.

use std::collections::HashMap;

use safetensors::{Dtype, SafeTensors};
use thiserror::Error;

use super::model::{Architecture, POLICY_PLANES, WDL_OUTPUTS};
use crate::encoding::board::{CASTLING_BITS, EN_PASSANT_VOCAB, PIECE_VOCAB};

#[derive(Debug, Error)]
pub enum WeightsError {
    #[error("model.safetensors is not valid: {0}")]
    Format(#[from] safetensors::SafeTensorError),
    #[error("tensor {name} is missing")]
    Missing { name: String },
    #[error("tensor {name} has shape {actual:?}, expected {expected:?}")]
    Shape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("tensor {name} has dtype {dtype:?}; only F32, F16 and BF16 are read")]
    Dtype { name: String, dtype: Dtype },
}

/// A row-major fp32 tensor.
#[derive(Debug, Clone, PartialEq)]
struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A `LayerNorm`'s affine parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Norm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// A `Linear`: `weight` is `[out, in]` row-major as PyTorch stores it.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    pub out_features: usize,
    pub in_features: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// One pre-norm encoder layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub norm1: Norm,
    /// Packed `[q; k; v]`: `[3·d_model, d_model]`.
    pub in_proj: Linear,
    pub out_proj: Linear,
    pub norm2: Norm,
    pub linear1: Linear,
    pub linear2: Linear,
}

/// Every parameter of a `board`-mode network.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    /// `[PIECE_VOCAB, d_model]`.
    pub piece_embed: Vec<f32>,
    /// `[64, d_model]`.
    pub square_embed: Vec<f32>,
    /// `[d_model]`.
    pub state_embed: Vec<f32>,
    /// `Linear(4 -> d_model)`.
    pub castling_proj: Linear,
    /// `[EN_PASSANT_VOCAB, d_model]`.
    pub en_passant_embed: Vec<f32>,
    pub layers: Vec<Layer>,
    pub final_norm: Norm,
    pub policy_norm: Norm,
    /// `Linear(d_model -> 73)`.
    pub policy_proj: Linear,
    pub value_norm: Norm,
    /// `Linear(d_model -> d_model)`, then GELU.
    pub value_ff1: Linear,
    /// `Linear(d_model -> 3)`.
    pub value_ff2: Linear,
}

impl Weights {
    /// Parse `model.safetensors` for `arch`. Reading the file is `net::parse`'s job, which
    /// bounds its size first.
    pub fn from_bytes(bytes: &[u8], arch: &Architecture) -> Result<Self, WeightsError> {
        let file = SafeTensors::deserialize(bytes)?;
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(file.len());
        for (name, view) in file.tensors() {
            tensors.insert(name.clone(), to_f32(&name, &view)?);
        }
        let mut reader = Reader { tensors };
        let d = arch.d_model;

        let mut layers = Vec::with_capacity(arch.n_layers);
        for i in 0..arch.n_layers {
            let p = format!("encoder.layers.{i}.");
            layers.push(Layer {
                norm1: reader.norm(&format!("{p}norm1"), d)?,
                in_proj: reader.linear_named(
                    &format!("{p}self_attn.in_proj_weight"),
                    &format!("{p}self_attn.in_proj_bias"),
                    3 * d,
                    d,
                )?,
                out_proj: reader.linear(&format!("{p}self_attn.out_proj"), d, d)?,
                norm2: reader.norm(&format!("{p}norm2"), d)?,
                linear1: reader.linear(&format!("{p}linear1"), arch.d_ff, d)?,
                linear2: reader.linear(&format!("{p}linear2"), d, arch.d_ff)?,
            });
        }
        Ok(Self {
            piece_embed: reader.take("piece_embed.weight", &[PIECE_VOCAB, d])?,
            square_embed: reader.take("square_embed.weight", &[64, d])?,
            state_embed: reader.take("state_embed", &[d])?,
            castling_proj: reader.linear("castling_proj", d, CASTLING_BITS as usize)?,
            en_passant_embed: reader.take("en_passant_embed.weight", &[EN_PASSANT_VOCAB, d])?,
            layers,
            final_norm: reader.norm("final_norm", d)?,
            policy_norm: reader.norm("policy_head.norm", d)?,
            policy_proj: reader.linear("policy_head.proj", POLICY_PLANES, d)?,
            value_norm: reader.norm("value_head.norm", d)?,
            value_ff1: reader.linear("value_head.ff.0", d, d)?,
            value_ff2: reader.linear("value_head.ff.2", WDL_OUTPUTS, d)?,
        })
    }
}

struct Reader {
    tensors: HashMap<String, Tensor>,
}

impl Reader {
    fn take(&mut self, name: &str, expected: &[usize]) -> Result<Vec<f32>, WeightsError> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| WeightsError::Missing {
                name: name.to_string(),
            })?;
        if tensor.shape != expected {
            return Err(WeightsError::Shape {
                name: name.to_string(),
                actual: tensor.shape,
                expected: expected.to_vec(),
            });
        }
        Ok(tensor.data)
    }

    fn norm(&mut self, prefix: &str, d: usize) -> Result<Norm, WeightsError> {
        Ok(Norm {
            weight: self.take(&format!("{prefix}.weight"), &[d])?,
            bias: self.take(&format!("{prefix}.bias"), &[d])?,
        })
    }

    fn linear(&mut self, prefix: &str, out: usize, inp: usize) -> Result<Linear, WeightsError> {
        self.linear_named(
            &format!("{prefix}.weight"),
            &format!("{prefix}.bias"),
            out,
            inp,
        )
    }

    fn linear_named(
        &mut self,
        weight: &str,
        bias: &str,
        out: usize,
        inp: usize,
    ) -> Result<Linear, WeightsError> {
        Ok(Linear {
            out_features: out,
            in_features: inp,
            weight: self.take(weight, &[out, inp])?,
            bias: self.take(bias, &[out])?,
        })
    }
}

fn to_f32(name: &str, view: &safetensors::tensor::TensorView<'_>) -> Result<Tensor, WeightsError> {
    let shape = view.shape().to_vec();
    let bytes = view.data();
    let data = match view.dtype() {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
            .collect(),
        Dtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect(),
        dtype => {
            return Err(WeightsError::Dtype {
                name: name.to_string(),
                dtype,
            });
        }
    };
    Ok(Tensor { shape, data })
}

/// IEEE half to single.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalise.
            let shift = frac.leading_zeros() - 21; // frac has 10 significant bits in a u32
            let frac = (frac << (shift + 1)) & 0x3ff;
            sign | ((113 - shift) << 23) | (frac << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// Single to IEEE half, round to nearest even.
#[cfg(test)]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        // Inf or NaN.
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow to inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow to zero
        }
        // Subnormal half.
        let frac = frac | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = frac >> shift;
        let rem = frac & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half + 1
        } else {
            half
        };
        return sign | rounded as u16;
    }
    let half = ((e as u32) << 10) | (frac >> 13);
    let rem = frac & 0x1fff;
    let rounded = if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half + 1
    } else {
        half
    };
    sign | rounded as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversions_round_trip() {
        for &x in &[
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            65504.0,
            6.1e-5,
            1e-7,
            std::f32::consts::PI,
            -2.5e-3,
        ] {
            let h = f32_to_f16(x);
            let back = f16_to_f32(h);
            let tol = (x.abs() * 1e-3).max(1e-7);
            assert!((back - x).abs() <= tol, "{x} -> {h:#x} -> {back}");
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f32_to_f16(1e10), 0x7c00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
    }
}
