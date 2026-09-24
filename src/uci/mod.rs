//! The UCI loop: stdin lines in, protocol lines out, for the life of the process.
//!
//! A reader thread turns stdin into [`Event::Line`]s; the search thread sends
//! [`Event::Search`]s; the loop below receives both on one channel and is the only code
//! that writes to the GUI. That keeps `isready`, `stop` and `quit` responsive during a
//! search and keeps every protocol line whole.

pub mod command;
pub mod output;

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::mpsc::{self, Sender};

use crate::diag::{self, Tee, Transcript};
use crate::engine::{Engine, Go, SearchEvent};
use crate::options::Options;
use command::Command;

pub const NAME: &str = "Marina";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");

/// Everything the loop reacts to.
#[derive(Debug)]
pub enum Event {
    /// One line from the GUI.
    Line(String),
    /// stdin closed: the GUI is gone, so behave as if it had sent `quit`.
    Eof,
    Search(SearchEvent),
}

/// Spawn the stdin reader and run the loop until `quit` or EOF.
pub fn run(engine: &mut Engine, out: impl Write) -> io::Result<()> {
    let (tx, rx) = mpsc::channel::<Event>();
    spawn_reader(tx.clone());
    let mut out = Tee::new(out, None);
    let mut transcript: Option<Arc<Transcript>> = None;
    let mut failed = false;
    let (search_tx, search_rx) = mpsc::channel::<SearchEvent>();
    // Forward search events onto the main channel so one receiver sees everything. The
    // thread ends when the last `SearchEvent` sender is gone: `search_tx` below and the
    // clone each search thread holds.
    let forward = tx.clone();
    let forwarder = std::thread::spawn(move || {
        for event in search_rx {
            if forward.send(Event::Search(event)).is_err() {
                break;
            }
        }
    });

    for event in rx.iter() {
        match event {
            Event::Eof => break,
            Event::Search(SearchEvent::Info(line)) => {
                writeln!(out, "{line}")?;
                out.flush()?;
            }
            Event::Search(SearchEvent::Failed(error)) => {
                // Like a network that fails to load: say why and quit after the pending
                // `bestmove`, rather than play on with whatever the tree held.
                output::info_string(&mut out, &format!("network: {error}; quitting"))?;
                failed = true;
            }
            Event::Search(SearchEvent::BestMove { best, ponder }) => {
                output::bestmove(&mut out, &best, ponder.as_deref())?;
                engine.search_finished();
                if failed {
                    out.flush()?;
                    std::process::exit(2);
                }
                if engine.has_pending() {
                    report_network(engine, &mut out)?;
                    engine.start_pending(search_tx.clone());
                }
            }
            Event::Line(line) => {
                if let Some(transcript) = transcript.as_ref() {
                    transcript.input(&line);
                }
                tracing::trace!(line, "received");
                match command::parse(&line) {
                    Command::Uci => {
                        output::id(&mut out, &format!("{NAME} {VERSION}"), AUTHORS)?;
                        for spec in Options::specs() {
                            output::option(&mut out, &spec)?;
                        }
                        output::uciok(&mut out)?;
                    }
                    Command::Debug(on) => diag::set_uci_debug(on),
                    Command::IsReady => {
                        report_network(engine, &mut out)?;
                        output::readyok(&mut out)?;
                    }
                    Command::SetOption { name, value } => {
                        match engine.options.set(&name, value.as_deref()) {
                            Ok(()) if name.eq_ignore_ascii_case("DebugLogFile") => {
                                transcript = open_transcript(&engine.options, &mut out)?;
                                out.set_transcript(transcript.clone());
                                diag::set_transcript(transcript.clone());
                            }
                            Ok(()) => tracing::debug!(option = %name, "set"),
                            Err(error) => output::info_string(&mut out, &error.to_string())?,
                        }
                    }
                    Command::Register => {} // no registration required
                    Command::UciNewGame => engine.new_game(),
                    Command::Position { fen, moves } => {
                        match crate::position::Game::from_uci(
                            fen.as_deref(),
                            &moves,
                            shakmaty::CastlingMode::Standard,
                        ) {
                            Ok(game) => {
                                tracing::debug!(ply = game.ply(), turn = ?game.turn(), "position");
                                engine.set_position(game);
                            }
                            Err(error) => output::info_string(&mut out, &error.to_string())?,
                        }
                    }
                    Command::Go(limits) => {
                        tracing::debug!(?limits, "go");
                        report_network(engine, &mut out)?;
                        match engine.go(limits, search_tx.clone()) {
                            Go::Started => {}
                            Go::Queued => tracing::debug!("go queued behind the running search"),
                            Go::Replaced => tracing::debug!("go replaced the queued one"),
                        }
                    }
                    Command::Stop => engine.stop(),
                    Command::PonderHit => engine.ponderhit(),
                    Command::Quit => break,
                    Command::Unknown(_) => {} // spec: ignore
                }
            }
        }
    }
    engine.shutdown();
    // The search thread has been joined, so it has sent everything it will, but its last
    // events may still be inside the forwarder. With the search thread gone `search_tx` is
    // the last sender: dropping it ends the forwarder's channel, and joining the forwarder
    // guarantees every event has reached `rx` before the final drain. The reader thread
    // keeps `tx` alive, so `rx` never disconnects and the drain stops at empty instead.
    drop(search_tx);
    let _ = forwarder.join();
    while let Ok(event) = rx.try_recv() {
        if let Event::Search(SearchEvent::BestMove { best, ponder }) = event {
            output::bestmove(&mut out, &best, ponder.as_deref())?;
        }
    }
    Ok(())
}

fn spawn_reader(tx: Sender<Event>) {
    std::thread::spawn(move || {
        // Bytes, not `lines()`: a GUI may send a path in a non-UTF-8 encoding, and one bad
        // byte must not end the session.
        let stdin = io::stdin();
        let mut stdin = stdin.lock();
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match stdin.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf)
                        .trim_end_matches(['\n', '\r'])
                        .to_string();
                    if tx.send(Event::Line(line)).is_err() {
                        return;
                    }
                }
            }
        }
        let _ = tx.send(Event::Eof);
    });
}

/// Load the network the options ask for and tell the GUI what happened.
fn report_network(engine: &mut Engine, out: &mut impl Write) -> io::Result<()> {
    match engine.ensure_network() {
        Ok(Some(description)) => output::info_string(out, &format!("network: {description}")),
        Ok(None) => Ok(()),
        Err(error) => {
            // A network that fails to load is never substituted: a tournament would record
            // garbage games as if they were ours. Say why and quit so the GUI or harness
            // registers an engine failure instead.
            output::info_string(out, &format!("network: {error}; quitting"))?;
            out.flush()?;
            std::process::exit(2);
        }
    }?;
    match engine.ensure_tablebase() {
        Ok(Some(description)) => output::info_string(out, &description),
        Ok(None) => Ok(()),
        Err(error) => output::info_string(out, &format!("Syzygy: {error}")),
    }
}

fn open_transcript(options: &Options, out: &mut impl Write) -> io::Result<Option<Arc<Transcript>>> {
    if options.debug_log_file.is_empty() {
        return Ok(None);
    }
    match Transcript::open(&options.debug_log_file) {
        Ok(opened) => Ok(Some(Arc::new(opened))),
        Err(error) => {
            output::info_string(
                out,
                &format!("DebugLogFile {}: {error}", options.debug_log_file),
            )?;
            Ok(None)
        }
    }
}
