pub mod cli;
pub mod diag;
pub mod encoding;
pub mod engine;
pub mod net;
pub mod options;
pub mod position;
pub mod search;
pub mod tablebase;
pub mod uci;
pub mod verify;

/// Entry point: the UCI engine on stdin/stdout until `quit` or EOF, or a subcommand.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Accelerate reads this once at start-up; see `net::gemm::single_threaded`.
    if cfg!(target_os = "macos") && std::env::var_os("VECLIB_MAXIMUM_THREADS").is_none() {
        // SAFETY: called before any other thread exists.
        unsafe { std::env::set_var("VECLIB_MAXIMUM_THREADS", "1") };
    }
    diag::init();
    match cli::parse() {
        cli::Command::Uci => {
            let mut engine = engine::Engine::new();
            // Not a held `StdoutLock`: `debug on` mirrors tracing events to stdout from
            // other threads, which must be able to take the lock per line.
            uci::run(&mut engine, std::io::stdout())?;
        }
        cli::Command::Bench(args) => cli::bench(args)?,
        cli::Command::Verify(args) => cli::verify(args)?,
    }
    Ok(())
}
