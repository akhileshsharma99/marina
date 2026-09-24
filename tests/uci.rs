//! Drive the built binary over stdin/stdout the way a GUI does.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

struct Engine {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Engine {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_marina"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn marina");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut engine = Self {
            child,
            stdin,
            lines,
        };
        // The protocol tests search with the small embedded network: the unoptimised test
        // build evaluates the default (`small`) too slowly for tests that only need a move.
        engine.send("setoption name Network value nano");
        engine
    }

    fn send(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// Raw bytes, for input that is not valid UTF-8.
    fn send_bytes(&mut self, bytes: &[u8]) {
        self.stdin.write_all(bytes).unwrap();
        self.stdin.flush().unwrap();
    }

    /// Next line within the timeout, or None.
    fn next(&self, timeout: Duration) -> Option<String> {
        self.lines.recv_timeout(timeout).ok()
    }

    /// Read until a line starting with `prefix`, returning everything read. The deadline
    /// is generous: the unoptimised test build evaluates the network slowly, and the tests
    /// run in parallel.
    fn expect(&self, prefix: &str) -> Vec<String> {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut seen = Vec::new();
        while Instant::now() < deadline {
            if let Some(line) = self.next(deadline - Instant::now()) {
                let done = line.starts_with(prefix);
                seen.push(line);
                if done {
                    return seen;
                }
            }
        }
        panic!("did not see {prefix:?}; saw {seen:?}");
    }

    fn wait(mut self) -> std::process::ExitStatus {
        drop(self.stdin);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "engine did not exit");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[test]
fn handshake_then_quit() {
    let mut engine = Engine::spawn();
    engine.send("uci");
    let lines = engine.expect("uciok");
    assert!(lines[0].starts_with("id name Marina "));
    assert_eq!(lines[1], "id author Akhilesh Sharma");
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("option name WeightsFile type string"))
    );
    assert!(lines.iter().any(
        |l| l.starts_with("option name Network type combo default small var ")
            && l.ends_with(" var file")
    ));
    assert!(
        lines
            .iter()
            .any(|l| l == "option name Ponder type check default false")
    );
    engine.send("isready");
    assert_eq!(engine.expect("readyok").last().unwrap(), "readyok");
    engine.send("quit");
    assert!(engine.wait().success());
}

#[test]
fn plays_a_legal_move_and_reports_bad_input() {
    let mut engine = Engine::spawn();
    engine.send("position startpos moves e2e4 e7e5");
    engine.send("go wtime 1000 btime 1000");
    let lines = engine.expect("bestmove");
    let info = lines
        .iter()
        .find(|l| l.starts_with("info depth "))
        .expect("an info line before bestmove");
    assert!(
        info.contains(" nodes ") && info.contains(" score cp ") && info.contains(" pv "),
        "{info}"
    );
    let best = lines.last().unwrap();
    assert!(best.starts_with("bestmove ") && best.len() >= 13, "{best}");
    engine.send("position startpos moves e2e4 e4e5");
    let lines = engine.expect("info string");
    assert!(
        lines
            .last()
            .unwrap()
            .contains("illegal or malformed move \"e4e5\"")
    );
    engine.send("quit");
    assert!(engine.wait().success());
}

/// A byte that is not UTF-8 (a GUI sending a path in a legacy encoding) must not end the
/// session: the line is read lossily and the engine carries on.
#[test]
fn a_non_utf8_byte_on_stdin_does_not_end_the_session() {
    let mut engine = Engine::spawn();
    engine.send_bytes(b"setoption name DebugLogFile value /nonexistent/l\xffog.txt\n");
    engine.send("isready");
    let lines = engine.expect("readyok");
    assert!(lines.last().unwrap() == "readyok", "{lines:?}");
    engine.send("position startpos");
    engine.send("go nodes 4");
    engine.expect("bestmove");
    engine.send("quit");
    assert!(engine.wait().success());
}

#[test]
fn infinite_search_answers_isready_and_stops_with_one_bestmove() {
    let mut engine = Engine::spawn();
    engine.send("position startpos");
    engine.send("go infinite");
    // Info lines are fine; no bestmove may arrive on its own.
    let quiet_until = Instant::now() + Duration::from_millis(300);
    while Instant::now() < quiet_until {
        if let Some(line) = engine.next(quiet_until - Instant::now()) {
            assert!(!line.starts_with("bestmove"), "{line}");
        }
    }
    engine.send("isready");
    assert_eq!(engine.expect("readyok").last().unwrap(), "readyok");
    engine.send("stop");
    let lines = engine.expect("bestmove");
    assert_eq!(
        lines.iter().filter(|l| l.starts_with("bestmove")).count(),
        1
    );
    // Exactly one bestmove overall; a second `stop` produces nothing.
    engine.send("stop");
    assert!(engine.next(Duration::from_millis(300)).is_none());
    engine.send("quit");
    assert!(engine.wait().success());
}

/// GUIs send `stop`, `position`, `go` back to back without waiting for the `bestmove`:
/// the new `go` must start once it arrives, for the new position, not be refused.
#[test]
fn go_right_after_stop_is_queued_and_the_latest_go_wins() {
    let mut engine = Engine::spawn();
    engine.send("position startpos");
    engine.send("go infinite");
    std::thread::sleep(Duration::from_millis(100));
    engine.send("stop\nposition startpos moves e2e4\ngo nodes 4");
    let first = engine.expect("bestmove");
    assert!(!first.iter().any(|l| l.contains("go ignored")), "{first:?}");
    let second = engine.expect("bestmove");
    // The second search answers 1.e4 with a Black move; its reports are its own.
    let reply = second.last().unwrap().split_whitespace().nth(1).unwrap();
    assert!(
        [
            "e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g8f6", "d7d6", "g7g6"
        ]
        .contains(&reply),
        "{second:?}"
    );
    assert!(
        second.iter().any(|l| l.starts_with("info depth ")),
        "{second:?}"
    );

    // A `stop` sent for the queued search stops it as soon as it starts: two bestmoves,
    // then silence.
    engine.send("go infinite");
    std::thread::sleep(Duration::from_millis(100));
    engine.send("stop\nposition startpos moves d2d4\ngo infinite\nstop");
    engine.expect("bestmove");
    engine.expect("bestmove");
    assert!(engine.next(Duration::from_millis(500)).is_none());
    // A burst of `stop`/`position`/`go` faster than the search winds down (an analysis
    // GUI stepping through moves): the latest `go` is the one that runs, for its own
    // position, and the superseded ones send no `bestmove`. Two bestmoves in total: the
    // running search's and the last go's.
    engine.send("go infinite");
    std::thread::sleep(Duration::from_millis(100));
    engine.send(
        "stop\nposition startpos moves e2e4\ngo infinite\nstop\nposition startpos moves d2d4\ngo infinite\nstop\nposition startpos moves c2c4\ngo nodes 4",
    );
    let first = engine.expect("bestmove");
    assert!(!first.iter().any(|l| l.contains("go ignored")), "{first:?}");
    let last = engine.expect("bestmove");
    let reply = last.last().unwrap().split_whitespace().nth(1).unwrap();
    // Black's reply to 1.c4: never a reply to 1.e4 or 1.d4 only.
    assert!(
        [
            "e7e5", "c7c5", "e7e6", "g8f6", "c7c6", "g7g6", "f7f5", "b7b6"
        ]
        .contains(&reply),
        "{last:?}"
    );
    assert!(engine.next(Duration::from_millis(500)).is_none());
    engine.send("quit");
    assert!(engine.wait().success());
}

#[test]
fn ponder_waits_for_ponderhit_or_stop() {
    let mut engine = Engine::spawn();
    // Pondering never answers on its own, not even a mate in one.
    engine.send("position fen 6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1");
    engine.send("go ponder wtime 1000 btime 1000");
    let quiet_until = Instant::now() + Duration::from_millis(300);
    while Instant::now() < quiet_until {
        if let Some(line) = engine.next(quiet_until - Instant::now()) {
            assert!(!line.starts_with("bestmove"), "{line}");
        }
    }
    engine.send("ponderhit");
    let lines = engine.expect("bestmove");
    assert_eq!(lines.last().unwrap(), "bestmove d1d8");
    assert!(
        lines.iter().any(|l| l.contains(" score mate 1 ")),
        "{lines:?}"
    );
    // A ponder miss: `stop` gets a bestmove, then the real position is searched afresh.
    engine.send("position startpos moves e2e4 e7e5");
    engine.send("go ponder wtime 1000 btime 1000");
    std::thread::sleep(Duration::from_millis(50));
    engine.send("stop");
    let lines = engine.expect("bestmove");
    assert_eq!(
        lines.iter().filter(|l| l.starts_with("bestmove")).count(),
        1
    );
    engine.send("position startpos moves e2e4 c7c5");
    engine.send("go wtime 1000 btime 1000");
    let lines = engine.expect("bestmove");
    assert!(lines.last().unwrap().starts_with("bestmove "));
    engine.send("quit");
    assert!(engine.wait().success());
}

#[test]
fn infinite_holds_a_proven_root_until_stop() {
    let mut engine = Engine::spawn();
    engine.send("position fen 6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1");
    engine.send("go infinite");
    let quiet_until = Instant::now() + Duration::from_millis(300);
    while Instant::now() < quiet_until {
        if let Some(line) = engine.next(quiet_until - Instant::now()) {
            assert!(!line.starts_with("bestmove"), "{line}");
        }
    }
    engine.send("stop");
    assert_eq!(engine.expect("bestmove").last().unwrap(), "bestmove d1d8");
    engine.send("quit");
    assert!(engine.wait().success());
}

#[test]
fn multipv_and_wdl_lines() {
    let mut engine = Engine::spawn();
    engine.send("setoption name MultiPV value 3");
    engine.send("setoption name UCI_ShowWDL value true");
    engine.send("position startpos");
    // A handful of nodes: the unoptimised test build evaluates the network slowly.
    engine.send("go nodes 4");
    let lines = engine.expect("bestmove");
    let best = lines
        .last()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .to_string();
    let report: Vec<&String> = lines
        .iter()
        .rev()
        .skip(1)
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    for (i, line) in report.iter().enumerate() {
        assert!(line.contains(&format!(" multipv {} ", i + 1)), "{line}");
        assert!(line.contains(" wdl "), "{line}");
    }
    let first_move = report[0]
        .split_whitespace()
        .skip_while(|&w| w != "pv")
        .nth(1)
        .unwrap();
    assert_eq!(first_move, best);
    engine.send("quit");
    assert!(engine.wait().success());
}

#[test]
fn quit_during_search_still_exits() {
    let mut engine = Engine::spawn();
    engine.send("go infinite");
    std::thread::sleep(Duration::from_millis(100));
    engine.send("quit");
    let status = engine.wait();
    assert!(status.success());
}

/// `quit` right behind `go`: the search is stopped and joined at `quit` and sends its
/// `bestmove` on the way out. Every search that starts owes one, and it must reach the GUI
/// rather than be lost between the search's channel and the loop's.
#[test]
fn quit_right_after_go_still_delivers_the_bestmove() {
    let mut engine = Engine::spawn();
    engine.send("position startpos\ngo nodes 50\nquit");
    let lines = engine.expect("bestmove");
    assert_eq!(
        lines.iter().filter(|l| l.starts_with("bestmove")).count(),
        1,
        "{lines:?}"
    );
    assert!(engine.wait().success());
}

/// The GUI switches networks while a search is running and searches again at once. The
/// running search, still on the old network, stores its tree when it finishes, which is
/// after the engine has forgotten the old tree on loading the new network; the first
/// search on the new network must not continue from it. In debug mode each search says
/// what it did with the tree it found.
#[test]
fn a_tree_from_the_previous_network_is_not_reused_after_a_switch() {
    let mut engine = Engine::spawn();
    engine.send("debug on");
    engine.send("setoption name Backend value cpu");
    engine.send("setoption name Network value small");
    engine.send("position startpos");
    engine.send("go infinite");
    std::thread::sleep(Duration::from_millis(300));
    // The switch, then `stop`, `position`, `go` back to back the way a GUI sends them: the
    // `go` arrives while the first search is winding down and is queued behind it.
    engine.send("setoption name Network value nano\nstop\nposition startpos\ngo nodes 200");
    let first = engine.expect("bestmove");
    let second = engine.expect("bestmove");
    assert!(
        first
            .iter()
            .chain(&second)
            .any(|l| l.contains("network: embedded nano")),
        "{first:?} {second:?}"
    );
    assert!(
        !second.iter().any(|l| l.contains("tree reused")),
        "{second:?}"
    );
    assert!(
        second
            .iter()
            .any(|l| l.contains("tree discarded") || l.contains("tree fresh")),
        "{second:?}"
    );
    let info = second
        .iter()
        .rev()
        .find(|l| l.starts_with("info depth "))
        .expect("an info line from the second search");
    let words: Vec<&str> = info.split_whitespace().collect();
    let at = words.iter().position(|&w| w == "nodes").unwrap();
    let nodes: u64 = words[at + 1].parse().unwrap();
    assert!(nodes <= 200, "{info}");
    engine.send("quit");
    assert!(engine.wait().success());
}

#[test]
fn eof_ends_the_process() {
    let mut engine = Engine::spawn();
    engine.send("uci");
    engine.expect("uciok");
    assert!(engine.wait().success());
}

/// The engine loads its network on `isready` and searches with it: the embedded default, or the
/// directory `MARINA_WEIGHTS` names when set.
#[test]
fn loads_weights_from_the_option_and_searches() {
    let mut engine = Engine::spawn();
    engine.send("uci");
    engine.expect("uciok");
    if let Ok(weights) = std::env::var("MARINA_WEIGHTS") {
        engine.send("setoption name Network value file");
        engine.send(&format!("setoption name WeightsFile value {weights}"));
    }
    engine.send("setoption name Backend value cpu");
    engine.send("isready");
    let lines = engine.expect("readyok");
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("info string network: ") && l.contains("cpu fp32")),
        "{lines:?}"
    );
    engine.send("position startpos moves e2e4");
    // A handful of nodes: the unoptimised test build evaluates the network slowly.
    engine.send("go nodes 4");
    let lines = engine.expect("bestmove");
    let best = lines.last().unwrap();
    assert!(best.starts_with("bestmove "), "{best}");
    // A trained network answers 1.e4 with a real move, not the fake's arbitrary one.
    let mv = best.split_whitespace().nth(1).unwrap();
    assert!(
        [
            "e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g8f6", "d7d6", "g7g6"
        ]
        .contains(&mv),
        "unexpected reply {mv}"
    );
    engine.send("quit");
    assert!(engine.wait().success());
}
