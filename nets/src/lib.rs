//! The networks compiled into Marina, and the names it knows.
//!
//! The engine's `nets.toml` is the source of truth; `build.rs` fetches the networks it marks `embed`,
//! checks their hashes, and generates [`DEFAULT`], [`KNOWN`] and [`EMBEDDED`]. A crate of
//! its own so the weights compile once: the engine's edit-build loop never touches them.

/// Search settings tuned for a network (`[nets.<name>.search]` in `nets.toml`): the
/// defaults of `CPuct`, `FPUReduction` and `Batch` while it is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchDefaults {
    pub cpuct_centi: u32,
    pub fpu_reduction_centi: u32,
    pub batch: u32,
}

/// A network the engine knows by name, embedded or not.
#[derive(Debug, Clone, Copy)]
pub struct Known {
    pub name: &'static str,
    /// First 8 hex digits of the weights' sha256; names the release asset.
    pub revision: &'static str,
    /// `embed = true` in `nets.toml`: compiled in when the `embedded` feature is on.
    pub embedded: bool,
    pub search: SearchDefaults,
}

/// A network compiled into this binary.
#[derive(Debug, Clone, Copy)]
pub struct Embedded {
    pub name: &'static str,
    pub revision: &'static str,
    pub model_json: &'static str,
    pub model_safetensors: &'static [u8],
    /// Path of the network's `vectors.npz` on the machine that built this binary: 2,048
    /// positions with the outputs of the network as trained, for `cargo test` and `verify`.
    pub vectors: &'static str,
    /// sha256 of `vectors.npz`, from `nets.toml`: what a copy fetched at run time must hash to.
    pub vectors_sha256: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/embedded_nets.rs"));

impl Embedded {
    /// Short provenance for GUIs and logs: `nano@7fc01ecb`.
    pub fn describe(&self) -> String {
        format!("{}@{}", self.name, self.revision)
    }

    /// The release asset name of one of this network's files: `nano-7fc01ecb.vectors.npz`.
    pub fn asset(&self, file: &str) -> String {
        format!("{}-{}.{file}", self.name, self.revision)
    }

    /// Where to download one of this network's files from.
    pub fn asset_url(&self, file: &str) -> String {
        format!("{RELEASE_URL}/{}", self.asset(file))
    }
}
