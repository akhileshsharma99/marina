//! Networks compiled into the binary and the names the engine knows.
//!
//! `nets.toml` is the source of truth; the `marina-nets` crate (`nets/`) fetches the networks
//! it marks `embed`, checks their hashes, and generates `DEFAULT`, `KNOWN` and `EMBEDDED`,
//! re-exported here. The `Network` option resolves
//! against them (`<embedded>` is [`DEFAULT`], a listed name is that network); a directory
//! comes through `WeightsFile`.

use thiserror::Error;

pub use marina_nets::{DEFAULT, EMBEDDED, Embedded, KNOWN, Known, SearchDefaults};

#[derive(Debug, Error)]
pub enum EmbeddedError {
    #[error(
        "network {name} is not built into this binary{reason}; download {name}-{revision}.model.json \
         and {name}-{revision}.model.safetensors from \
         https://github.com/akhileshsharma99/marina/releases/tag/nets into a directory as \
         model.json and model.safetensors, and set WeightsFile to it"
    )]
    NotEmbedded {
        name: &'static str,
        revision: &'static str,
        /// Empty, or " (built without the `embedded` feature)" when it would otherwise be.
        reason: &'static str,
    },
    #[error("{0} is not a known network name or a directory")]
    Unknown(String),
}

/// The network `<embedded>` means.
pub fn default() -> Result<&'static Embedded, EmbeddedError> {
    find(DEFAULT)
}

/// The embedded network called `name`, or why it is not available.
pub fn find(name: &str) -> Result<&'static Embedded, EmbeddedError> {
    if let Some(net) = EMBEDDED.iter().find(|net| net.name == name) {
        return Ok(net);
    }
    match KNOWN.iter().find(|known| known.name == name) {
        Some(known) => Err(EmbeddedError::NotEmbedded {
            name: known.name,
            revision: known.revision,
            reason: if known.embedded && !cfg!(feature = "embedded") {
                " (built without the `embedded` feature)"
            } else {
                ""
            },
        }),
        None => Err(EmbeddedError::Unknown(name.to_string())),
    }
}

/// Whether `spec` names a known network (as opposed to a directory).
pub fn is_known(spec: &str) -> bool {
    KNOWN.iter().any(|known| known.name == spec)
}

/// The `nets.toml` entry for a network name, if there is one.
pub fn known(name: &str) -> Option<&'static Known> {
    KNOWN.iter().find(|known| known.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_consistent() {
        assert!(is_known(DEFAULT));
        for net in EMBEDDED {
            assert!(is_known(net.name));
            assert!(!net.model_json.is_empty());
            assert!(!net.model_safetensors.is_empty());
            assert!(net.describe().starts_with(net.name));
        }
        assert!(!is_known("/some/dir"));
    }

    #[cfg(feature = "embedded")]
    #[test]
    fn default_network_is_embedded_and_parses() {
        let net = default().expect("default embedded network");
        let arch = crate::net::Architecture::from_json(net.model_json).expect("model.json");
        let weights = crate::net::weights::Weights::from_bytes(net.model_safetensors, &arch)
            .expect("model.safetensors");
        assert_eq!(weights.layers.len(), arch.n_layers);
    }
}
