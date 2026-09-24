//! The engine's UCI options: what `uci` declares and what `setoption` changes.
//!
//! Names are matched case-insensitively as the specification requires. Values are parsed
//! into typed fields and validated against the declared ranges; an invalid `setoption` is
//! reported and leaves the option unchanged. Floats (`CPuct`, `FPUReduction`) are declared as
//! `spin` options in hundredths, as GUIs have no float type.

use strum::{Display, EnumIter, EnumString, IntoEnumIterator};
use thiserror::Error;

use crate::net::embedded::SearchDefaults;

use crate::uci::output::{OptionSpec, OptionType};

/// Spec meaning "the default network compiled into the binary" (`marina verify --weights`,
/// [`crate::net::parse`]); the UCI options pick networks with `Network`.
pub const EMBEDDED_WEIGHTS: &str = "<embedded>";

/// Value of `Network` meaning "the directory `WeightsFile` names".
pub const NETWORK_FILE: &str = "file";

/// The `Network` combo's choices: every embedded network, then `file`. GUIs give a
/// `*File` option a file picker, so a name has to be a dropdown to be selectable at all.
fn network_vars() -> &'static [&'static str] {
    static VARS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    VARS.get_or_init(|| {
        // The default first, then the rest as listed, then `file`.
        let default = crate::net::embedded::DEFAULT;
        let mut vars: Vec<&'static str> = vec![default];
        vars.extend(
            crate::net::embedded::KNOWN
                .iter()
                .map(|k| k.name)
                .filter(|&n| n != default),
        );
        vars.push(NETWORK_FILE);
        vars
    })
}

/// How `score cp` is derived from the value head's expected score (`ScoreType`). Neither
/// changes the search or the moves; `UCI_ShowWDL` prints the underlying probabilities.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, EnumString, Display, EnumIter, clap::ValueEnum,
)]
#[strum(serialize_all = "lowercase", ascii_case_insensitive)]
pub enum ScoreType {
    /// The classic logistic relation between centipawns and expected score,
    /// `cp = 400 log10(s / (1 - s))`: +100 is about 64% expected score, +200 about 76%.
    #[default]
    Centipawn,
    /// Lc0's mapping, `90 tan(1.5637 q)`: smaller in the middle, explosive near ±1.
    Lc0,
}

/// Arithmetic of the network's linear layers (`Precision`). `Auto` is the fastest the
/// loaded backend supports: fp8 on CUDA with an Ada or newer card and a calibrated
/// network, else fp16 there; fp16 on Metal; fp32 on the CPU. An explicit value a backend
/// cannot do falls back to the nearest it can and says so; the network line reports what
/// runs.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, EnumString, Display, EnumIter, clap::ValueEnum,
)]
#[strum(serialize_all = "lowercase", ascii_case_insensitive)]
pub enum Precision {
    #[default]
    Auto,
    Fp32,
    Fp16,
    Fp8,
}

/// Where the network runs. `Auto` is resolved when the network is loaded: CUDA when this
/// build has the backend and a device answers, otherwise the CPU.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, EnumString, Display, EnumIter, clap::ValueEnum,
)]
#[strum(serialize_all = "lowercase", ascii_case_insensitive)]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
}

/// Current option values. The defaults are what the engine plays with.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// `Network`: an embedded network's name, or [`NETWORK_FILE`] for `WeightsFile`.
    pub network: String,
    /// `WeightsFile`: a directory holding `model.json` + `model.safetensors`, read only
    /// when `Network` is [`NETWORK_FILE`].
    pub weights_file: String,
    pub backend: Backend,
    /// Threads the CPU backend's forward pass runs on. The search itself is one gather
    /// thread plus one network thread; the CUDA backend ignores this.
    pub threads: u32,
    /// Leaves collected per network call; unset, the selected network's tuned value.
    pub batch: Option<u32>,
    /// Batch ramp: a batch asks for at most `tree size / batch_ramp` leaves so a small tree
    /// is not swamped by one batch; 0 turns the ramp off.
    pub batch_ramp: u32,
    /// Collision budget as a percentage of the batch: stop filling a batch once the visits
    /// it could not place exceed this; 0 means no limit.
    pub collision_budget_percent: u32,
    /// Batches queued at the network at once: 1 evaluates one batch at a time, 2 gathers
    /// the next batch while the previous one is being evaluated (hiding the network's
    /// latency, at the price of `Batch` more leaves chosen without feedback).
    pub in_flight: u32,
    /// Arithmetic of the linear layers; see [`Precision`].
    pub precision: Precision,
    /// Syzygy tablebase directories separated by the platform path separator; empty = none.
    pub syzygy_path: String,
    /// Probe positions with at most this many pieces.
    pub syzygy_probe_limit: u32,
    /// Per-move slack for the GUI round trip, in milliseconds.
    pub move_overhead_ms: u32,
    /// Root moves to report an `info` line for; the search is the same whatever the value.
    pub multipv: u32,
    /// Report `wdl` (win/draw/loss in permille) on every `info` line.
    pub show_wdl: bool,
    /// How `score cp` is computed from the value head.
    pub score_type: ScoreType,
    /// PUCT exploration constant, in hundredths; unset, the selected network's tuned value.
    pub cpuct_centi: Option<u32>,
    /// First-play urgency reduction, in hundredths: an unvisited child is assumed to be
    /// worth its parent's value minus this times the square root of the explored prior
    /// mass; unset, the selected network's tuned value.
    pub fpu_reduction_centi: Option<u32>,
    /// Append every protocol line, in and out, to this file; empty = off.
    pub debug_log_file: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            network: crate::net::embedded::DEFAULT.to_string(),
            weights_file: String::new(),
            backend: Backend::Auto,
            threads: DEFAULTS.threads,
            batch: None,
            batch_ramp: DEFAULTS.batch_ramp,
            collision_budget_percent: DEFAULTS.collision_budget_percent,
            in_flight: DEFAULTS.in_flight,
            precision: Precision::Auto,
            syzygy_path: String::new(),
            syzygy_probe_limit: DEFAULTS.syzygy_probe_limit,
            move_overhead_ms: DEFAULTS.move_overhead_ms,
            multipv: DEFAULTS.multipv,
            show_wdl: false,
            score_type: ScoreType::Centipawn,
            cpuct_centi: None,
            fpu_reduction_centi: None,
            debug_log_file: String::new(),
        }
    }
}

/// Why a `setoption` was rejected. The option keeps its previous value.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum OptionError {
    #[error("unknown option {0:?}")]
    Unknown(String),
    #[error("option {name} requires a value")]
    MissingValue { name: &'static str },
    #[error("option {name}: {value:?} is not a valid value")]
    Invalid { name: &'static str, value: String },
    #[error("option {name}: {value} must be between {min} and {max}")]
    OutOfRange {
        name: &'static str,
        value: i64,
        min: i64,
        max: i64,
    },
}

/// Every option the engine declares, in the order printed after `id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
enum Name {
    Network,
    WeightsFile,
    Backend,
    Threads,
    Batch,
    BatchRamp,
    CollisionBudget,
    InFlight,
    Precision,
    SyzygyPath,
    SyzygyProbeLimit,
    MoveOverhead,
    CPuct,
    FpuReduction,
    Ponder,
    MultiPv,
    ShowWdl,
    ScoreType,
    DebugLogFile,
}

impl Name {
    /// The name as declared to the GUI.
    fn as_str(self) -> &'static str {
        match self {
            Name::Network => "Network",
            Name::WeightsFile => "WeightsFile",
            Name::Backend => "Backend",
            Name::Threads => "Threads",
            Name::Batch => "Batch",
            Name::BatchRamp => "BatchRamp",
            Name::InFlight => "InFlight",
            Name::Precision => "Precision",
            Name::CollisionBudget => "CollisionBudget",
            Name::SyzygyPath => "SyzygyPath",
            Name::SyzygyProbeLimit => "SyzygyProbeLimit",
            Name::MoveOverhead => "MoveOverhead",
            Name::CPuct => "CPuct",
            Name::FpuReduction => "FPUReduction",
            Name::Ponder => "Ponder",
            Name::MultiPv => "MultiPV",
            Name::ShowWdl => "UCI_ShowWDL",
            Name::ScoreType => "ScoreType",
            Name::DebugLogFile => "DebugLogFile",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Name::iter().find(|candidate| candidate.as_str().eq_ignore_ascii_case(name.trim()))
    }

    fn spec(self) -> OptionSpec<'static> {
        let defaults = DEFAULTS;
        // The declared defaults of the per-network options are the default network's.
        let search = Options::default().search_defaults();
        let kind = match self {
            Name::Network => OptionType::Combo {
                default: crate::net::embedded::DEFAULT,
                vars: network_vars(),
            },
            Name::WeightsFile => OptionType::String { default: "" },
            Name::Backend => OptionType::Combo {
                default: "auto",
                vars: BACKENDS,
            },
            Name::Threads => spin(defaults.threads, THREADS),
            Name::Batch => spin(search.batch, BATCH),
            Name::BatchRamp => spin(defaults.batch_ramp, BATCH_RAMP),
            Name::CollisionBudget => spin(defaults.collision_budget_percent, COLLISION_BUDGET),
            Name::InFlight => spin(defaults.in_flight, IN_FLIGHT),
            Name::Precision => OptionType::Combo {
                default: "auto",
                vars: PRECISIONS,
            },
            Name::SyzygyPath => OptionType::String { default: "" },
            Name::SyzygyProbeLimit => spin(defaults.syzygy_probe_limit, SYZYGY_PROBE_LIMIT),
            Name::MoveOverhead => spin(defaults.move_overhead_ms, MOVE_OVERHEAD),
            Name::CPuct => spin(search.cpuct_centi, CPUCT),
            Name::FpuReduction => spin(search.fpu_reduction_centi, FPU_REDUCTION),
            Name::Ponder => OptionType::Check { default: false },
            Name::MultiPv => spin(defaults.multipv, MULTIPV),
            Name::ShowWdl => OptionType::Check { default: false },
            Name::ScoreType => OptionType::Combo {
                default: "centipawn",
                vars: SCORE_TYPES,
            },
            Name::DebugLogFile => OptionType::String { default: "" },
        };
        OptionSpec {
            name: self.as_str(),
            kind,
        }
    }
}

const SCORE_TYPES: &[&str] = &["centipawn", "lc0"];
const PRECISIONS: &[&str] = &["auto", "fp32", "fp16", "fp8"];

/// Backends this build can offer.
#[cfg(all(feature = "cuda", target_os = "macos"))]
const BACKENDS: &[&str] = &["auto", "cpu", "cuda", "metal"];
#[cfg(all(feature = "cuda", not(target_os = "macos")))]
const BACKENDS: &[&str] = &["auto", "cpu", "cuda"];
#[cfg(all(not(feature = "cuda"), target_os = "macos"))]
const BACKENDS: &[&str] = &["auto", "cpu", "metal"];
#[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
const BACKENDS: &[&str] = &["auto", "cpu"];

/// Numeric defaults, shared by the declarations and [`Options::default`]. `CPuct`,
/// `FPUReduction` and `Batch` come from the selected network's `nets.toml` entry
/// ([`Options::search_defaults`]); [`FALLBACK_SEARCH`] serves a directory network.
const DEFAULTS: Defaults = Defaults {
    threads: 1,
    batch_ramp: 4,
    collision_budget_percent: 100,
    in_flight: 2,
    syzygy_probe_limit: 5,
    move_overhead_ms: 20,
    multipv: 1,
};

/// Search settings for a network `nets.toml` does not know (a directory).
const FALLBACK_SEARCH: SearchDefaults = SearchDefaults {
    cpuct_centi: 175,
    fpu_reduction_centi: 25,
    batch: 128,
};

struct Defaults {
    threads: u32,
    batch_ramp: u32,
    collision_budget_percent: u32,
    in_flight: u32,
    syzygy_probe_limit: u32,
    move_overhead_ms: u32,
    multipv: u32,
}

const THREADS: (u32, u32) = (1, 256);
const BATCH: (u32, u32) = (1, 1024);
const BATCH_RAMP: (u32, u32) = (0, 64);
const COLLISION_BUDGET: (u32, u32) = (0, 1000);
const IN_FLIGHT: (u32, u32) = (1, 8);
const SYZYGY_PROBE_LIMIT: (u32, u32) = (0, 7);
const MOVE_OVERHEAD: (u32, u32) = (0, 5000);
const MULTIPV: (u32, u32) = (1, 256);
const CPUCT: (u32, u32) = (1, 1000);
const FPU_REDUCTION: (u32, u32) = (0, 100);

fn spin(default: u32, (min, max): (u32, u32)) -> OptionType<'static> {
    OptionType::Spin {
        default: i64::from(default),
        min: i64::from(min),
        max: i64::from(max),
    }
}

impl Options {
    /// The `option name ...` declarations to print after `id`.
    pub fn specs() -> Vec<OptionSpec<'static>> {
        Name::iter().map(Name::spec).collect()
    }

    /// Apply `setoption name <name> [value <value>]`.
    pub fn set(&mut self, name: &str, value: Option<&str>) -> Result<(), OptionError> {
        let option = Name::parse(name).ok_or_else(|| OptionError::Unknown(name.to_string()))?;
        let key = option.as_str();
        let value = value.map(str::trim);
        match option {
            Name::Network => {
                let wanted = string_value(value).unwrap_or(crate::net::embedded::DEFAULT);
                let var = network_vars()
                    .iter()
                    .find(|v| v.eq_ignore_ascii_case(wanted))
                    .ok_or_else(|| OptionError::Invalid {
                        name: key,
                        value: wanted.to_string(),
                    })?;
                self.network = (*var).to_string();
            }
            Name::WeightsFile => {
                self.weights_file = string_value(value).unwrap_or("").to_string();
            }
            Name::SyzygyPath => {
                self.syzygy_path = string_value(value).unwrap_or("").to_string();
            }
            Name::DebugLogFile => {
                self.debug_log_file = string_value(value).unwrap_or("").to_string();
            }
            Name::Backend => self.backend = parse_enum(key, value)?,
            Name::Threads => self.threads = parse_spin(key, value, THREADS)?,
            Name::Batch => self.batch = Some(parse_spin(key, value, BATCH)?),
            Name::BatchRamp => self.batch_ramp = parse_spin(key, value, BATCH_RAMP)?,
            Name::CollisionBudget => {
                self.collision_budget_percent = parse_spin(key, value, COLLISION_BUDGET)?;
            }
            Name::InFlight => self.in_flight = parse_spin(key, value, IN_FLIGHT)?,
            Name::Precision => self.precision = parse_enum(key, value)?,
            Name::SyzygyProbeLimit => {
                self.syzygy_probe_limit = parse_spin(key, value, SYZYGY_PROBE_LIMIT)?;
            }
            Name::MoveOverhead => self.move_overhead_ms = parse_spin(key, value, MOVE_OVERHEAD)?,
            Name::CPuct => self.cpuct_centi = Some(parse_spin(key, value, CPUCT)?),
            Name::FpuReduction => {
                self.fpu_reduction_centi = Some(parse_spin(key, value, FPU_REDUCTION)?);
            }
            // Declared so GUIs offer pondering (they only send `go ponder` to engines that
            // list it); the value is the GUI's to act on, the search is the same either way.
            Name::Ponder => {
                parse_check(key, value)?;
            }
            Name::MultiPv => self.multipv = parse_spin(key, value, MULTIPV)?,
            Name::ShowWdl => self.show_wdl = parse_check(key, value)?,
            Name::ScoreType => self.score_type = parse_enum(key, value)?,
        }
        Ok(())
    }

    /// The tuned search settings of the selected network: the defaults of `CPuct`,
    /// `FPUReduction` and `Batch` while nothing else is set.
    pub fn search_defaults(&self) -> SearchDefaults {
        crate::net::embedded::known(&self.weights_spec())
            .map_or(FALLBACK_SEARCH, |known| known.search)
    }

    /// What to load, as [`crate::net::resolve`] takes it: the embedded network `Network`
    /// names, or the `WeightsFile` directory when `Network` is `file` (empty: the default
    /// network, and the label says so).
    pub fn weights_spec(&self) -> String {
        if self.network != NETWORK_FILE {
            return self.network.clone();
        }
        let file = self.weights_file.trim();
        if file.is_empty() {
            return crate::net::embedded::DEFAULT.to_string();
        }
        file.to_string()
    }

    /// PUCT exploration constant.
    pub fn cpuct(&self) -> f32 {
        self.cpuct_centi
            .unwrap_or_else(|| self.search_defaults().cpuct_centi) as f32
            / 100.0
    }

    /// FPU reduction.
    pub fn fpu_reduction(&self) -> f32 {
        self.fpu_reduction_centi
            .unwrap_or_else(|| self.search_defaults().fpu_reduction_centi) as f32
            / 100.0
    }

    /// Leaves collected per network call.
    pub fn batch(&self) -> u32 {
        self.batch.unwrap_or_else(|| self.search_defaults().batch)
    }

    /// True when the network comes from the binary rather than a directory.
    pub fn uses_embedded_weights(&self) -> bool {
        crate::net::embedded::is_known(&self.weights_spec())
    }

    /// How to name the network in messages: `embedded small@73bc41fb` or the directory.
    pub fn weights_label(&self) -> String {
        let spec = self.weights_spec();
        match crate::net::embedded::find(&spec) {
            Ok(net) if self.network == NETWORK_FILE && self.weights_file.trim().is_empty() => {
                format!(
                    "embedded {} (Network is file but WeightsFile is empty)",
                    net.describe()
                )
            }
            Ok(net) => format!("embedded {}", net.describe()),
            Err(_) => spec,
        }
    }
}

/// String options: `<empty>` (the spec's spelling of an empty string) and a missing value
/// both mean empty.
fn string_value(value: Option<&str>) -> Option<&str> {
    match value {
        None | Some("") | Some("<empty>") => None,
        Some(other) => Some(other),
    }
}

fn parse_enum<T: std::str::FromStr>(
    name: &'static str,
    value: Option<&str>,
) -> Result<T, OptionError> {
    let raw = value.ok_or(OptionError::MissingValue { name })?;
    raw.parse().map_err(|_| OptionError::Invalid {
        name,
        value: raw.to_string(),
    })
}

fn parse_spin(
    name: &'static str,
    value: Option<&str>,
    (min, max): (u32, u32),
) -> Result<u32, OptionError> {
    let raw = value.ok_or(OptionError::MissingValue { name })?;
    let parsed: i64 = raw.parse().map_err(|_| OptionError::Invalid {
        name,
        value: raw.to_string(),
    })?;
    if parsed < i64::from(min) || parsed > i64::from(max) {
        return Err(OptionError::OutOfRange {
            name,
            value: parsed,
            min: i64::from(min),
            max: i64::from(max),
        });
    }
    Ok(parsed as u32)
}

fn parse_check(name: &'static str, value: Option<&str>) -> Result<bool, OptionError> {
    match value.map(|v| v.to_ascii_lowercase()).as_deref() {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(other) => Err(OptionError::Invalid {
            name,
            value: other.to_string(),
        }),
        None => Err(OptionError::MissingValue { name }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declarations_cover_every_option_with_the_default_values() {
        let specs = Options::specs();
        let rendered: Vec<String> = specs.iter().map(OptionSpec::render).collect();
        assert_eq!(rendered.len(), Name::iter().count());
        assert!(
            rendered.contains(&"option name WeightsFile type string default <empty>".to_string())
        );
        assert!(rendered.iter().any(|line| {
            line.starts_with("option name Network type combo default small var ")
                && line.ends_with(" var file")
        }));
        assert!(rendered.iter().any(|line| {
            line.starts_with("option name Backend type combo default auto var auto var cpu")
        }));
        assert!(
            rendered.contains(&"option name Threads type spin default 1 min 1 max 256".to_string())
        );
        assert!(
            rendered.contains(&"option name SyzygyPath type string default <empty>".to_string())
        );
        assert!(
            rendered
                .contains(&"option name CPuct type spin default 175 min 1 max 1000".to_string())
        );
        assert!(rendered.contains(&"option name Ponder type check default false".to_string()));
        assert!(
            rendered.contains(&"option name MultiPV type spin default 1 min 1 max 256".to_string())
        );
        assert!(rendered.contains(&"option name UCI_ShowWDL type check default false".to_string()));
        // The declared defaults are the struct defaults.
        let defaults = Options::default();
        assert_eq!(defaults.cpuct(), 1.75);
        assert_eq!(defaults.fpu_reduction(), 0.25);
        assert!(defaults.uses_embedded_weights());
    }

    #[test]
    fn search_defaults_follow_the_selected_network_until_set() {
        let mut options = Options::default();
        let small = crate::net::embedded::known("small").unwrap().search;
        let nano = crate::net::embedded::known("nano").unwrap().search;
        assert_ne!(
            small.cpuct_centi, nano.cpuct_centi,
            "the test needs two different pins"
        );
        assert_eq!(options.cpuct(), small.cpuct_centi as f32 / 100.0);
        options.set("Network", Some("nano")).unwrap();
        assert_eq!(options.cpuct(), nano.cpuct_centi as f32 / 100.0);
        assert_eq!(options.batch(), nano.batch);
        // An explicit value sticks across network changes.
        options.set("CPuct", Some("123")).unwrap();
        options.set("Network", Some("small")).unwrap();
        assert_eq!(options.cpuct(), 1.23);
        // A directory network gets the fallback; WeightsFile is ignored unless Network is file.
        options.set("WeightsFile", Some("/some/dir")).unwrap();
        assert_eq!(options.cpuct(), 1.23);
        assert_eq!(options.batch(), small.batch);
        options.set("Network", Some("file")).unwrap();
        assert_eq!(options.batch(), FALLBACK_SEARCH.batch);
        assert!(options.set("Network", Some("medium")).is_err());
    }

    #[test]
    fn set_is_case_insensitive_and_typed() {
        let mut options = Options::default();
        options.set("threads", Some("8")).unwrap();
        assert_eq!(options.threads, 8);
        options.set("BACKEND", Some("CPU")).unwrap();
        assert_eq!(options.backend, Backend::Cpu);
        options.set("UCI_ShowWDL", Some("True")).unwrap();
        assert!(options.show_wdl);
        options.set("CPuct", Some("200")).unwrap();
        assert_eq!(options.cpuct(), 2.0);
        options.set("Network", Some("FILE")).unwrap();
        options
            .set("WeightsFile", Some("/nets/marina-192x6"))
            .unwrap();
        assert_eq!(options.weights_spec(), "/nets/marina-192x6");
        assert!(!options.uses_embedded_weights());
        options.set("WeightsFile", Some("<empty>")).unwrap();
        assert!(options.uses_embedded_weights()); // file + nothing named: the default
        options.set("Network", Some("<empty>")).unwrap();
        assert_eq!(options.weights_spec(), crate::net::embedded::DEFAULT);
        options.set("SyzygyPath", Some("/tb/3-4-5:/tb/6")).unwrap();
        assert_eq!(options.syzygy_path, "/tb/3-4-5:/tb/6");
        options.set("BatchRamp", Some("0")).unwrap();
        assert_eq!(options.batch_ramp, 0);
        options.set("CollisionBudget", Some("250")).unwrap();
        assert_eq!(options.collision_budget_percent, 250);
        options.set("MultiPV", Some("4")).unwrap();
        assert_eq!(options.multipv, 4);
        options.set("UCI_ShowWDL", Some("true")).unwrap();
        assert!(options.show_wdl);
        options.set("Ponder", Some("true")).unwrap();
        assert_eq!(
            options.set("Ponder", Some("maybe")),
            Err(OptionError::Invalid {
                name: "Ponder",
                value: "maybe".into()
            })
        );
    }

    #[test]
    fn invalid_values_are_rejected_and_leave_the_option_unchanged() {
        let mut options = Options::default();
        assert_eq!(
            options.set("Threads", Some("0")),
            Err(OptionError::OutOfRange {
                name: "Threads",
                value: 0,
                min: 1,
                max: 256
            })
        );
        assert_eq!(options.threads, 1);
        assert_eq!(
            options.set("Threads", Some("many")),
            Err(OptionError::Invalid {
                name: "Threads",
                value: "many".into()
            })
        );
        assert_eq!(
            options.set("Backend", Some("tpu")),
            Err(OptionError::Invalid {
                name: "Backend",
                value: "tpu".into()
            })
        );
        assert_eq!(options.backend, Backend::Auto);
        assert_eq!(
            options.set("Batch", None),
            Err(OptionError::MissingValue { name: "Batch" })
        );
        assert_eq!(
            options.set("UCI_ShowWDL", Some("yes")),
            Err(OptionError::Invalid {
                name: "UCI_ShowWDL",
                value: "yes".into()
            })
        );
        assert_eq!(
            options.set("Hash", Some("16")),
            Err(OptionError::Unknown("Hash".into()))
        );
        assert_eq!(options, Options::default());
    }
}
