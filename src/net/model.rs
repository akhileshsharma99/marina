//! The architecture card (`model.json`) that ships next to `model.safetensors`.
//!
//! The card describes the network completely: input encoding, token layout, encoder
//! shape and the heads. The engine reads only the fields that parametrise the forward
//! pass; everything else is documentation for humans and is checked lightly (format and
//! version) so an incompatible export fails at load time rather than at play time.

use serde::Deserialize;
use thiserror::Error;

/// `format` the card must declare.
pub const FORMAT: &str = "marina-weights";
/// `format_version` this engine understands.
pub const FORMAT_VERSION: u32 = 1;

/// Tokens the encoder sees per position in `board` mode: one state token, 64 squares.
pub const BOARD_TOKENS: usize = 65;
/// Move planes per square token in the policy head.
pub const POLICY_PLANES: usize = 73;
/// Win, draw, loss.
pub const WDL_OUTPUTS: usize = 3;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model.json is not valid: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "model.json format is {format:?} version {version}; this engine reads {FORMAT:?} version {FORMAT_VERSION}"
    )]
    Format { format: String, version: u32 },
    #[error("context_mode {0:?} is not supported by this engine (only \"board\")")]
    ContextMode(String),
    #[error("d_model {d_model} is not divisible by n_heads {n_heads}")]
    Heads { d_model: usize, n_heads: usize },
    #[error(
        "model.json describes a degenerate network (a zero dimension or a non-positive layer_norm_eps)"
    )]
    Degenerate,
    #[error(
        "tokens {tokens} with square_token_start {square_token_start}: board mode needs one \
         or more state tokens followed by exactly 64 square tokens"
    )]
    Tokens {
        tokens: usize,
        square_token_start: usize,
    },
    #[error(
        "encoder activation {0:?} is not supported (\"gelu (exact, erf)\" or \"gelu (tanh approximation)\")"
    )]
    Activation(String),
    #[error(
        "encoder.fp8.activation_amax must hold four positive maxima for each of the {0} layers"
    )]
    Fp8Amax(usize),
}

/// Which GELU the feed-forward blocks were trained with. The two differ by < 1e-3; the
/// tanh form is the one cuBLASLt can fuse into the FFN1 GEMM epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gelu {
    /// `0.5·x·(1 + erf(x/√2))`, `torch.nn.GELU()`.
    Erf,
    /// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, `torch.nn.GELU(approximate="tanh")`.
    Tanh,
}

impl Gelu {
    fn parse(card: &str) -> Option<Self> {
        let s = card.trim().to_ascii_lowercase();
        if !s.starts_with("gelu") {
            return None;
        }
        if s.contains("tanh") {
            Some(Self::Tanh)
        } else if s == "gelu" || s.contains("erf") || s.contains("exact") {
            Some(Self::Erf)
        } else {
            None
        }
    }
}

/// What the forward pass needs to know.
#[derive(Debug, Clone, PartialEq)]
pub struct Architecture {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub layer_norm_eps: f32,
    pub gelu: Gelu,
    /// Per layer, the largest |value| over the golden positions of the four tensors an FP8
    /// backend quantises with static scales: norm1 output, attention output, norm2 output,
    /// activated hidden state. Absent from cards exported before FP8 was supported.
    pub fp8_amax: Option<Vec<[f32; 4]>>,
    /// Encoder tokens per position.
    pub tokens: usize,
    /// Index of the first square token.
    pub square_token_start: usize,
    pub parameters: u64,
    /// Free-form provenance from the card (`checkpoint`), for `info string`.
    pub checkpoint: serde_json::Value,
}

impl Architecture {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Parse the text of `model.json`. Reading the file is `net::parse`'s job, which bounds
    /// its size first.
    pub fn from_json(text: &str) -> Result<Self, ModelError> {
        let card: Card = serde_json::from_str(text)?;
        if card.format != FORMAT || card.format_version != FORMAT_VERSION {
            return Err(ModelError::Format {
                format: card.format,
                version: card.format_version,
            });
        }
        if card.model.context_mode != "board" {
            return Err(ModelError::ContextMode(card.model.context_mode));
        }
        if card.model.d_model == 0
            || card.model.n_heads == 0
            || card.model.n_layers == 0
            || card.model.d_ff == 0
            || card.encoder.layer_norm_eps.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        {
            return Err(ModelError::Degenerate);
        }
        if !card.model.d_model.is_multiple_of(card.model.n_heads) {
            return Err(ModelError::Heads {
                d_model: card.model.d_model,
                n_heads: card.model.n_heads,
            });
        }
        // The kernels and the CPU pass index square tokens as `token - square_token_start`
        // into the 64-byte board, so the card must describe exactly that layout.
        // Bounded as well as consistent: the token count sizes every activation buffer, and
        // a wrapped or absurd value must fail here, not in a kernel.
        const MAX_SQUARE_TOKEN_START: usize = 64;
        if card.tokens.square_token_start < 1
            || card.tokens.square_token_start > MAX_SQUARE_TOKEN_START
            || card
                .tokens
                .square_token_start
                .checked_add(64)
                .is_none_or(|expected| expected != card.tokens.count)
        {
            return Err(ModelError::Tokens {
                tokens: card.tokens.count,
                square_token_start: card.tokens.square_token_start,
            });
        }
        let gelu = Gelu::parse(&card.encoder.activation)
            .ok_or_else(|| ModelError::Activation(card.encoder.activation.clone()))?;
        let fp8_amax = match card.encoder.fp8 {
            None => None,
            Some(fp8) => {
                let rows = fp8.activation_amax;
                if rows.len() != card.model.n_layers
                    || rows.iter().flatten().any(|&v| !(v > 0.0 && v.is_finite()))
                {
                    return Err(ModelError::Fp8Amax(card.model.n_layers));
                }
                Some(rows)
            }
        };
        Ok(Self {
            d_model: card.model.d_model,
            n_layers: card.model.n_layers,
            n_heads: card.model.n_heads,
            d_ff: card.model.d_ff,
            layer_norm_eps: card.encoder.layer_norm_eps,
            gelu,
            fp8_amax,
            tokens: card.tokens.count,
            square_token_start: card.tokens.square_token_start,
            parameters: card.model.parameters,
            checkpoint: card.checkpoint,
        })
    }

    /// One line for `info string`.
    pub fn describe(&self) -> String {
        let step = self
            .checkpoint
            .get("step")
            .and_then(serde_json::Value::as_u64)
            .map(|s| format!(" step {s}"))
            .unwrap_or_default();
        let parameters = if self.parameters > 0 {
            format!(", {:.2}M parameters", self.parameters as f64 / 1e6)
        } else {
            String::new()
        };
        format!(
            "{}x{} ({} heads, d_ff {}{parameters}{step})",
            self.d_model, self.n_layers, self.n_heads, self.d_ff
        )
    }
}

#[derive(Deserialize)]
struct Card {
    format: String,
    format_version: u32,
    model: CardModel,
    #[serde(default)]
    checkpoint: serde_json::Value,
    tokens: CardTokens,
    encoder: CardEncoder,
}

#[derive(Deserialize)]
struct CardModel {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_ff: usize,
    context_mode: String,
    #[serde(default)]
    parameters: u64,
}

#[derive(Deserialize)]
struct CardTokens {
    count: usize,
    square_token_start: usize,
}

#[derive(Deserialize)]
struct CardEncoder {
    layer_norm_eps: f32,
    activation: String,
    fp8: Option<CardFp8>,
}

#[derive(Deserialize)]
struct CardFp8 {
    activation_amax: Vec<[f32; 4]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const CARD: &str = r#"{
        "format": "marina-weights", "format_version": 1,
        "model": {"d_model": 192, "n_layers": 6, "n_heads": 6, "d_ff": 768, "dropout": 0.1,
                  "context_mode": "board", "parameters": 2739724},
        "checkpoint": {"step": 2400000},
        "inputs": {},
        "tokens": {"count": 65, "state_token_index": 0, "square_token_start": 1},
        "encoder": {"layers": 6, "layer_norm_eps": 1e-5, "activation": "gelu (exact, erf)"},
        "outputs": {}
    }"#;

    #[test]
    fn parses_the_card() {
        let arch = Architecture::from_json(CARD).unwrap();
        assert_eq!(arch.d_model, 192);
        assert_eq!(arch.head_dim(), 32);
        assert_eq!(arch.tokens, BOARD_TOKENS);
        assert_eq!(arch.layer_norm_eps, 1e-5);
        assert_eq!(arch.gelu, Gelu::Erf);
        assert_eq!(
            arch.describe(),
            "192x6 (6 heads, d_ff 768, 2.74M parameters step 2400000)"
        );
    }

    #[test]
    fn reads_the_fp8_calibration_when_present() {
        assert_eq!(Architecture::from_json(CARD).unwrap().fp8_amax, None);
        let rows: Vec<String> = (0..6)
            .map(|i| format!("[{}, 2.5, 3.0, 8.0]", i + 1))
            .collect();
        let with = CARD.replace(
            "\"activation\": \"gelu (exact, erf)\"",
            &format!(
                "\"activation\": \"gelu (exact, erf)\", \"fp8\": {{\"activation_amax\": [{}]}}",
                rows.join(", ")
            ),
        );
        let arch = Architecture::from_json(&with).unwrap();
        assert_eq!(arch.fp8_amax.as_ref().unwrap()[5], [6.0, 2.5, 3.0, 8.0]);
        let short = with.replace(", [6, 2.5, 3.0, 8.0]", "");
        assert!(matches!(
            Architecture::from_json(&short),
            Err(ModelError::Fp8Amax(6))
        ));
    }

    #[test]
    fn reads_the_activation() {
        let tanh = CARD.replace("gelu (exact, erf)", "gelu (tanh approximation)");
        assert_eq!(Architecture::from_json(&tanh).unwrap().gelu, Gelu::Tanh);
        let relu = CARD.replace("gelu (exact, erf)", "relu");
        assert!(matches!(
            Architecture::from_json(&relu),
            Err(ModelError::Activation(a)) if a == "relu"
        ));
    }

    #[test]
    fn rejects_token_layouts_that_wrap_or_exceed_the_bound() {
        // `square_token_start + 64` wrapping to `count` must not pass.
        let wrapped = CARD.replace("\"count\": 65", "\"count\": 0").replace(
            "\"square_token_start\": 1",
            "\"square_token_start\": 18446744073709551552",
        );
        assert!(matches!(
            Architecture::from_json(&wrapped),
            Err(ModelError::Tokens { tokens: 0, .. })
        ));
        // Consistent but absurd: the token count sizes every buffer.
        let huge = CARD.replace("\"count\": 65", "\"count\": 200064").replace(
            "\"square_token_start\": 1",
            "\"square_token_start\": 200000",
        );
        assert!(matches!(
            Architecture::from_json(&huge),
            Err(ModelError::Tokens { tokens: 200064, .. })
        ));
    }

    #[test]
    fn rejects_other_formats_and_modes() {
        let wrong = CARD.replace("\"format_version\": 1", "\"format_version\": 2");
        assert!(matches!(
            Architecture::from_json(&wrong),
            Err(ModelError::Format { version: 2, .. })
        ));
        let history = CARD.replace(
            "\"context_mode\": \"board\"",
            "\"context_mode\": \"history\"",
        );
        assert!(matches!(
            Architecture::from_json(&history),
            Err(ModelError::ContextMode(mode)) if mode == "history"
        ));
    }
}
