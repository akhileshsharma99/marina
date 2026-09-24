//! Command line: plain `marina` is the UCI engine on stdin/stdout; the `bench` and `verify`
//! subcommands are tools that do not speak UCI. Their results go to stdout as the output of
//! the command; diagnostics go through [`crate::diag`] like everywhere else.

use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "marina",
    version,
    about = "UCI chess engine running policy/value networks"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Measure search throughput, with a fake network by default or `--weights` for a real one.
    Bench(BenchArgs),
    /// Check a network backend against the golden vectors that ship with the weights.
    Verify(VerifyArgs),
}

/// What to run.
pub enum Command {
    /// The UCI engine on stdin/stdout.
    Uci,
    Bench(BenchArgs),
    Verify(VerifyArgs),
}

/// Parse the process arguments (exits with usage on error, as clap does).
pub fn parse() -> Command {
    match Cli::parse().command {
        None => Command::Uci,
        Some(Cmd::Bench(args)) => Command::Bench(args),
        Some(Cmd::Verify(args)) => Command::Verify(args),
    }
}

#[derive(clap::Args)]
pub struct VerifyArgs {
    /// Network to check: an embedded name (`nano`, `small`) or a directory with model.json
    /// and model.safetensors.
    #[arg(long)]
    pub weights: String,
    /// Golden vectors; defaults to vectors.npz next to the weights when they are a
    /// directory, and to the vectors fetched at build time for an embedded name.
    #[arg(long)]
    pub vectors: Option<std::path::PathBuf>,
    /// Backend to check.
    #[arg(long, value_enum, default_value_t = crate::options::Backend::Auto)]
    pub backend: crate::options::Backend,
    /// Arithmetic of the linear layers (the `Precision` option).
    #[arg(long, value_enum, default_value_t = crate::options::Precision::Auto)]
    pub precision: crate::options::Precision,
    /// Only the first N positions.
    #[arg(long)]
    pub limit: Option<usize>,
    /// Positions per network call.
    #[arg(long, default_value_t = 64)]
    pub batch: u32,
    /// Timed passes over the set after one warm-up pass; the rate is the mean.
    #[arg(long, default_value_t = 1)]
    pub repeat: usize,
}

/// The golden vectors of an embedded network: the copy the build fetched when this is the
/// machine that built the binary, else a copy in the user's cache, downloaded from the
/// networks' release on first use (they are 35 MB and not embedded).
fn embedded_vectors(
    net: &crate::net::embedded::Embedded,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let built = std::path::Path::new(net.vectors);
    if built.is_file() {
        return Ok(built.to_path_buf());
    }
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".cache")))
        .ok_or("no HOME to cache the golden vectors in; pass --vectors")?
        .join("marina");
    let path = cache.join(net.asset("vectors.npz"));
    if path.is_file() {
        if sha256_matches(&path, net.vectors_sha256)? {
            return Ok(path);
        }
        eprintln!(
            "cached golden vectors at {} do not hash as nets.toml says; fetching again",
            path.display()
        );
        std::fs::remove_file(&path)?;
    }
    let url = net.asset_url("vectors.npz");
    std::fs::create_dir_all(&cache)?;
    eprintln!(
        "fetching the golden vectors for {} (35 MB) to {}",
        net.describe(),
        path.display()
    );
    let partial = path.with_extension("npz.part");
    let status = std::process::Command::new("curl")
        .args(["-fsSL", &url, "-o"])
        .arg(&partial)
        .status()
        .map_err(|e| format!("cannot run curl ({e}); download {url} and pass --vectors"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&partial);
        return Err(
            format!("downloading {url} failed; download it yourself and pass --vectors").into(),
        );
    }
    if !sha256_matches(&partial, net.vectors_sha256)? {
        let _ = std::fs::remove_file(&partial);
        return Err(format!(
            "{url} does not hash to the sha256 nets.toml pins for {}; download it yourself, check it, and pass --vectors",
            net.describe()
        )
        .into());
    }
    std::fs::rename(&partial, &path)?;
    Ok(path)
}

/// Whether `path` hashes to `expected` (hex), reading it in blocks.
fn sha256_matches(path: &std::path::Path, expected: &str) -> std::io::Result<bool> {
    use sha2::Digest;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()).eq_ignore_ascii_case(expected))
}

/// Run `verify` and print the report; the error carries the failure when tolerances are
/// exceeded so the process exits non-zero.
pub fn verify(args: VerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vectors = match args.vectors {
        Some(path) => path,
        None if crate::net::embedded::is_known(&args.weights) => {
            embedded_vectors(crate::net::embedded::find(&args.weights)?)?
        }
        None => std::path::Path::new(&args.weights).join("vectors.npz"),
    };
    let started = std::time::Instant::now();
    let (report, tol, describe, evaluating) = crate::verify::run(
        &args.weights,
        &vectors,
        args.backend,
        args.precision,
        args.limit,
        args.batch as usize,
        args.repeat,
    )?;
    let total = started.elapsed();
    println!("network: {describe}");
    println!(
        "positions {}  evaluate {:.3}s per pass ({:.0} positions/s at batch {})  load+warmup {:.2}s",
        report.positions,
        evaluating.as_secs_f64(),
        report.positions as f64 / evaluating.as_secs_f64().max(1e-9),
        args.batch as usize,
        (total - evaluating * args.repeat.max(1) as u32).as_secs_f64()
    );
    println!(
        "max |diff|  logits {:.3e} (tol {:.0e})  priors {:.3e} (tol {:.0e})  wdl {:.3e} (tol {:.0e})",
        report.max_logit_diff,
        tol.logits,
        report.max_prior_diff,
        tol.priors,
        report.max_wdl_diff,
        tol.wdl
    );
    println!("best-move mismatches: {}", report.argmax_mismatches);
    if let Some((index, fen)) = &report.worst_position {
        println!("worst position: #{index} {fen}");
    }
    if report.passes(&tol) {
        println!("PASS");
        Ok(())
    } else {
        Err("FAIL: differences exceed tolerance".into())
    }
}

#[derive(clap::Args)]
pub struct BenchArgs {
    /// Simulations to run.
    #[arg(long, default_value_t = 100_000)]
    pub nodes: u64,
    /// Leaves per network call.
    #[arg(long, default_value_t = 256)]
    pub batch: u32,
    /// Which fake network to search with (ignored when --weights is given).
    #[arg(long, value_enum, default_value_t = FakeNet::Uniform)]
    pub net: FakeNet,
    /// Search with a real network instead: an embedded name (`nano`, `small`) or a directory
    /// with model.json and model.safetensors.
    #[arg(long)]
    pub weights: Option<String>,
    /// Backend for --weights.
    #[arg(long, value_enum, default_value_t = crate::options::Backend::Auto)]
    pub backend: crate::options::Backend,
    /// Arithmetic of the linear layers (the `Precision` option).
    #[arg(long, value_enum, default_value_t = crate::options::Precision::Auto)]
    pub precision: crate::options::Precision,
    /// Position to search; the start position by default.
    #[arg(long)]
    pub fen: Option<String>,
    /// Repeat and report each run.
    #[arg(long, default_value_t = 3)]
    pub runs: u32,
    /// Batches in flight at the network (1 = sequential).
    #[arg(long, default_value_t = crate::search::IN_FLIGHT)]
    pub in_flight: u32,
    /// Simulated network latency per batch, in microseconds (a GPU stand-in).
    #[arg(long, default_value_t = 0)]
    pub gpu_us: u64,
    /// Simulated network latency per leaf, in microseconds.
    #[arg(long, default_value_t = 0)]
    pub gpu_leaf_us: u64,
    /// Batch ramp divisor (UCI `BatchRamp`); 0 = off.
    #[arg(long, default_value_t = 4)]
    pub ramp: u32,
    /// Collision budget in percent of the batch (UCI `CollisionBudget`); 0 = unlimited.
    #[arg(long, default_value_t = 100)]
    pub collision_budget: u32,
    /// Syzygy directories (UCI `SyzygyPath`) to probe at the leaves.
    #[arg(long)]
    pub syzygy: Option<String>,
    /// Probe positions with at most this many pieces (UCI `SyzygyProbeLimit`).
    #[arg(long, default_value_t = 5)]
    pub syzygy_probe_limit: u32,
    /// Search this many nodes first and continue from that tree (tree reuse), so the timed
    /// runs measure a search that starts with a populated root.
    #[arg(long, default_value_t = 0)]
    pub warm: u64,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum FakeNet {
    /// Equal priors, value 0: flat trees, pure tree cost.
    Uniform,
    /// Deterministic pseudo-random priors: realistically shaped trees.
    Hash,
}

/// Run the tree benchmark and print one line per run.
pub fn bench(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::net::{HashNetwork, LatencyNetwork, Network, UniformNetwork};
    use crate::search::{Limits, Params, StopSignal};

    let BenchArgs {
        nodes,
        batch,
        net,
        weights,
        backend,
        precision,
        fen,
        runs,
        in_flight,
        gpu_us,
        gpu_leaf_us,
        ramp,
        collision_budget,
        syzygy,
        syzygy_probe_limit,
        warm,
    } = args;
    let fen = fen.as_deref();
    let tablebase = syzygy
        .as_deref()
        .map(|paths| crate::tablebase::Tablebase::open(paths, syzygy_probe_limit))
        .transpose()?;
    if let Some(tablebase) = &tablebase {
        println!("{}", tablebase.describe());
    }
    let gpu_per_batch = Duration::from_micros(gpu_us);
    let gpu_per_leaf = Duration::from_micros(gpu_leaf_us);
    let game = crate::position::Game::from_uci(fen, &[], shakmaty::CastlingMode::Standard)?;
    let inner: Box<dyn Network> = match weights {
        Some(spec) => crate::net::resolve(
            &spec,
            backend,
            crate::net::Device {
                threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
                max_batch: batch as usize,
                precision,
            },
        )?,
        None => match net {
            FakeNet::Uniform => Box::new(UniformNetwork),
            FakeNet::Hash => Box::new(HashNetwork::default()),
        },
    };
    let network: Box<dyn Network> = if gpu_per_batch.is_zero() && gpu_per_leaf.is_zero() {
        inner
    } else {
        Box::new(LatencyNetwork {
            inner,
            per_batch: gpu_per_batch,
            per_leaf: gpu_per_leaf,
        })
    };
    let params = Params {
        cpuct: 1.5,
        fpu_reduction: 0.25,
        batch,
        in_flight,
        batch_ramp: ramp,
        collision_budget_percent: collision_budget,
        move_overhead: Duration::ZERO,
        multipv: 1,
        show_wdl: false,
        score_type: crate::options::ScoreType::Centipawn,
    };
    let limits = Limits {
        nodes: Some(nodes),
        depth: None,
        time: None,
        until_stopped: false,
        ponder: false,
    };
    println!("{}", network.describe());
    for run in 1..=runs {
        let stop = StopSignal::new();
        let warm_tree = (warm > 0).then(|| {
            let warm_limits = Limits {
                nodes: Some(warm),
                ..limits
            };
            crate::search::search_with_tree(
                &game,
                &warm_limits,
                None,
                &params,
                network.as_ref(),
                tablebase.as_ref(),
                None,
                &stop,
                &mut |_| {},
            )
            .1
            .reroot(&[])
            .expect("a searched root is expanded")
        });
        let (summary, _) = crate::search::search_with_tree(
            &game,
            &limits,
            None,
            &params,
            network.as_ref(),
            tablebase.as_ref(),
            warm_tree,
            &stop,
            &mut |_| {},
        );
        let per_leaf_us = summary.elapsed.as_secs_f64() * 1e6 / summary.nodes.max(1) as f64;
        println!(
            "run {run}: nodes {} batch {batch} ramp {ramp} budget {collision_budget}% in-flight {in_flight} time {:.3}s  {:.1}K leaves/s  {per_leaf_us:.2} us/leaf  \
             batches {} ({:.0} leaves, {:.1} walks each) lost {} ({:.1}%)  tbhits {}  reused {}  depth avg {:.1} max {}  stop {:?}",
            summary.nodes,
            summary.elapsed.as_secs_f64(),
            summary.nps() / 1e3,
            summary.batches,
            summary.network_evals as f64 / summary.batches.max(1) as f64,
            summary.walks as f64 / summary.batches.max(1) as f64,
            summary.collisions,
            100.0 * summary.collisions as f64 / (summary.nodes + summary.collisions).max(1) as f64,
            summary.tbhits,
            summary.reused,
            summary.avg_depth,
            summary.max_depth,
            summary.stop_reason,
        );
    }
    Ok(())
}
