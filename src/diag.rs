//! Diagnostics: developer logging, UCI debug mode, and the protocol transcript.
//!
//! Three channels, each with one job:
//!
//! * **Developer logging** goes to stderr through [`tracing`]. Use `tracing::debug!`,
//!   `info!`, `warn!`, `error!` anywhere in the crate; the level is set by the `MARINA_LOG`
//!   environment variable (`warn` by default, e.g. `MARINA_LOG=debug` or
//!   `MARINA_LOG=marina::search=trace`). GUIs ignore stderr, so this can never corrupt the
//!   protocol.
//! * **UCI debug mode** (`debug on`) mirrors events at `debug` level and above to the GUI
//!   as `info string` lines, which is what the specification means by debug mode. It is a
//!   process-wide flag toggled by [`set_uci_debug`].
//! * **The transcript** ([`Transcript`]) records every protocol line, in and out, with
//!   `>>` (from the GUI) and `<<` (to the GUI) prefixes and a timestamp, to the file named
//!   by the `DebugLogFile` option. Wrap stdout in a [`Tee`] so everything written to the GUI
//!   is also written to the transcript; log input lines with [`Transcript::input`].
//!
//! Call [`init`] once at startup before anything logs.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use tracing::Level;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Environment variable selecting the stderr log level and targets.
pub const LOG_ENV: &str = "MARINA_LOG";

static UCI_DEBUG: AtomicBool = AtomicBool::new(false);
static STARTED: OnceLock<Instant> = OnceLock::new();
static TRANSCRIPT: Mutex<Option<Arc<Transcript>>> = Mutex::new(None);

/// Install the tracing subscriber: stderr at the `MARINA_LOG` level, plus `info string`
/// mirroring of `debug`-and-above events while UCI debug mode is on. Safe to call once;
/// later calls are ignored.
pub fn init() {
    STARTED.get_or_init(Instant::now);
    let stderr_filter = EnvFilter::try_from_env(LOG_ENV).unwrap_or_else(|_| EnvFilter::new("warn"));
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(io::stderr)
        .with_ansi(false)
        .with_target(true)
        .with_filter(stderr_filter);
    let uci_layer = tracing_subscriber::fmt::layer()
        .with_writer(InfoStringWriter)
        .with_ansi(false)
        .with_target(false)
        .with_level(true)
        .without_time()
        .with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
            uci_debug() && *metadata.level() <= Level::DEBUG
        }));
    // `try_init` so tests and embedders that already installed a subscriber keep theirs.
    let _ = tracing_subscriber::registry()
        .with(stderr_layer)
        .with(uci_layer)
        .try_init();
}

/// Turn UCI debug mode on or off (`debug on` / `debug off`).
pub fn set_uci_debug(enabled: bool) {
    UCI_DEBUG.store(enabled, Ordering::Relaxed);
    tracing::debug!(enabled, "uci debug mode");
}

/// Whether UCI debug mode is on.
pub fn uci_debug() -> bool {
    UCI_DEBUG.load(Ordering::Relaxed)
}

/// Install (or clear) the process-wide transcript used by the `info string` mirror, so
/// debug lines it writes to the GUI are recorded too. The UCI loop's [`Tee`] records the
/// protocol lines it writes itself.
pub fn set_transcript(transcript: Option<Arc<Transcript>>) {
    if let Ok(mut slot) = TRANSCRIPT.lock() {
        *slot = transcript;
    }
}

fn current_transcript() -> Option<Arc<Transcript>> {
    TRANSCRIPT.lock().ok().and_then(|slot| slot.clone())
}

/// Milliseconds since [`init`], for transcript timestamps.
fn elapsed_ms() -> u128 {
    STARTED.get().map_or(0, |start| start.elapsed().as_millis())
}

/// Writer that turns each formatted tracing event into one `info string` line on stdout.
/// Every event is written with a single locked `write_all`, so it cannot interleave with a
/// protocol line written by another thread.
struct InfoStringWriter;

impl<'a> MakeWriter<'a> for InfoStringWriter {
    type Writer = InfoStringLine;

    fn make_writer(&'a self) -> Self::Writer {
        InfoStringLine(Vec::new())
    }
}

struct InfoStringLine(Vec<u8>);

impl Write for InfoStringLine {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A formatted event may span several lines (fields, spans); each becomes its own
/// `info string` so the GUI never sees a bare continuation line.
fn info_string_lines(text: &str) -> String {
    let mut line = String::with_capacity(text.len() + 16);
    for part in text.lines().filter(|part| !part.trim().is_empty()) {
        line.push_str("info string ");
        line.push_str(part.trim_end());
        line.push('\n');
    }
    line
}

impl Drop for InfoStringLine {
    fn drop(&mut self) {
        let line = info_string_lines(&String::from_utf8_lossy(&self.0));
        if !line.is_empty() {
            let mut out = io::stdout().lock();
            let _ = out.write_all(line.as_bytes());
            let _ = out.flush();
            if let Some(transcript) = current_transcript() {
                for part in line.lines() {
                    transcript.output(part);
                }
            }
        }
    }
}

/// Append-only protocol transcript (`DebugLogFile`).
pub struct Transcript {
    file: Mutex<File>,
}

impl Transcript {
    /// Open (append) the transcript file.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    /// Record a line received from the GUI.
    pub fn input(&self, line: &str) {
        self.record(">>", line);
    }

    /// Record a line sent to the GUI.
    pub fn output(&self, line: &str) {
        self.record("<<", line);
    }

    fn record(&self, direction: &str, line: &str) {
        if let Ok(mut file) = self.file.lock() {
            let _ = writeln!(
                file,
                "[{:>8} ms] {direction} {}",
                elapsed_ms(),
                line.trim_end()
            );
        }
    }
}

/// A writer that forwards to the GUI and copies every completed line to a [`Transcript`].
pub struct Tee<W: Write> {
    inner: W,
    transcript: Option<Arc<Transcript>>,
    pending: Vec<u8>,
}

impl<W: Write> Tee<W> {
    pub fn new(inner: W, transcript: Option<Arc<Transcript>>) -> Self {
        Self {
            inner,
            transcript,
            pending: Vec::new(),
        }
    }

    /// Replace the transcript (for example after `setoption name DebugLogFile`).
    pub fn set_transcript(&mut self, transcript: Option<Arc<Transcript>>) {
        self.transcript = transcript;
        self.pending.clear();
    }

    fn record_lines(&mut self) {
        let Some(transcript) = self.transcript.as_ref() else {
            self.pending.clear();
            return;
        };
        while let Some(end) = self.pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=end).collect();
            transcript.output(&String::from_utf8_lossy(&line));
        }
    }
}

impl<W: Write> Write for Tee<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buf)?;
        if self.transcript.is_some() {
            self.pending.extend_from_slice(&buf[..written]);
            self.record_lines();
        }
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uci_debug_flag_round_trips() {
        set_uci_debug(true);
        assert!(uci_debug());
        set_uci_debug(false);
        assert!(!uci_debug());
    }

    #[test]
    fn tee_forwards_and_records_complete_lines() {
        let dir = std::env::temp_dir().join(format!("marina-transcript-{}", std::process::id()));
        let transcript = Arc::new(Transcript::open(&dir).unwrap());
        let mut sink: Vec<u8> = Vec::new();
        {
            let mut tee = Tee::new(&mut sink, Some(Arc::clone(&transcript)));
            tee.write_all(b"id name Marina\nuci").unwrap();
            tee.write_all(b"ok\n").unwrap();
            tee.flush().unwrap();
        }
        transcript.input("isready");
        assert_eq!(sink, b"id name Marina\nuciok\n");
        let text = std::fs::read_to_string(&dir).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "{text}");
        assert!(lines[0].ends_with("<< id name Marina"), "{}", lines[0]);
        assert!(lines[1].ends_with("<< uciok"), "{}", lines[1]);
        assert!(lines[2].ends_with(">> isready"), "{}", lines[2]);
        std::fs::remove_file(&dir).unwrap();
    }

    #[test]
    fn tee_without_transcript_is_a_plain_writer() {
        let mut sink: Vec<u8> = Vec::new();
        let mut tee = Tee::new(&mut sink, None);
        tee.write_all(b"readyok\n").unwrap();
        assert!(tee.pending.is_empty());
        drop(tee);
        assert_eq!(sink, b"readyok\n");
    }

    #[test]
    fn info_string_writer_prefixes_every_line() {
        assert_eq!(
            info_string_lines("DEBUG uci debug mode enabled=true\n\nsecond\n"),
            "info string DEBUG uci debug mode enabled=true\ninfo string second\n"
        );
        assert_eq!(info_string_lines("\n  \n"), "");
    }
}
