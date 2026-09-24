//! Engine-to-GUI messages of the UCI protocol.
//!
//! Specification: <https://backscattering.de/chess/uci/>, "Engine to GUI". Every message is
//! one line; every writer here appends the newline and flushes, because a GUI reads the
//! engine's stdout line by line and a `bestmove` left in a buffer looks like a hung engine.
//! In UCI mode nothing else writes to stdout except `diag`'s `debug on` mirror; anything
//! that is not protocol goes out through [`info_string`].

use std::fmt::{self, Write as _};
use std::io::{self, Write};

/// `id name <x>` and `id author <x>`; sent after `uci`, before the options.
pub fn id(out: &mut impl Write, name: &str, author: &str) -> io::Result<()> {
    writeln!(out, "id name {name}")?;
    writeln!(out, "id author {author}")?;
    out.flush()
}

/// `uciok`: identity and options are complete.
pub fn uciok(out: &mut impl Write) -> io::Result<()> {
    line(out, "uciok")
}

/// `readyok`: all input so far has been processed.
pub fn readyok(out: &mut impl Write) -> io::Result<()> {
    line(out, "readyok")
}

/// `bestmove <move> [ponder <move>]`; required exactly once per `go`, in UCI move notation
/// (`e7e8q` for promotions, `e1g1` for castling).
pub fn bestmove(out: &mut impl Write, best: &str, ponder: Option<&str>) -> io::Result<()> {
    match ponder {
        Some(ponder) => writeln!(out, "bestmove {best} ponder {ponder}")?,
        None => writeln!(out, "bestmove {best}")?,
    }
    out.flush()
}

/// `info string <text>`: the only channel for free text (warnings, errors, diagnostics).
pub fn info_string(out: &mut impl Write, text: &str) -> io::Result<()> {
    writeln!(out, "info string {text}")?;
    out.flush()
}

/// `score` of an `info` line, from the engine's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Score {
    /// Centipawns.
    Cp(i32),
    /// Mate in this many *moves* (not plies); negative when the engine is being mated.
    Mate(i32),
}

/// Whether a score is exact or only a bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bound {
    #[default]
    Exact,
    Lower,
    Upper,
}

/// Fields of one `info` line: the complete set the specification defines, so the renderer
/// is the protocol's, not the search's (which fills depth, seldepth, score, time, nodes,
/// nps, tbhits and pv). Only the fields that are `Some` (or non-empty) are written, in the
/// specification's order. `seldepth` requires `depth`
/// and `pv` should come with `time`; the writer does not enforce either.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Info<'a> {
    /// Search depth in plies.
    pub depth: Option<u32>,
    /// Selective depth in plies.
    pub seldepth: Option<u32>,
    /// Index of this line in MultiPV mode, starting at 1.
    pub multipv: Option<u32>,
    pub score: Option<Score>,
    pub bound: Bound,
    /// Win/draw/loss in permille from the engine's point of view (`UCI_ShowWDL` extension,
    /// used by Stockfish and Lc0). Written right after `score`.
    pub wdl: Option<(u32, u32, u32)>,
    /// Time searched in milliseconds.
    pub time: Option<u64>,
    pub nodes: Option<u64>,
    /// Nodes per second.
    pub nps: Option<u64>,
    /// Hash fullness in permille.
    pub hashfull: Option<u32>,
    /// Tablebase hits.
    pub tbhits: Option<u64>,
    /// Shredder-base hits.
    pub sbhits: Option<u64>,
    /// CPU load in permille.
    pub cpuload: Option<u32>,
    /// Move currently being searched at the root.
    pub currmove: Option<&'a str>,
    /// Index of `currmove` at the root, starting at 1.
    pub currmovenumber: Option<u32>,
    /// `refutation <move> [<reply> ...]`: only when `UCI_ShowRefutations` is on.
    pub refutation: &'a [&'a str],
    /// `currline [cpunr] <moves...>`: only when `UCI_ShowCurrLine` is on.
    pub currline: Option<(Option<u32>, &'a [&'a str])>,
    /// Principal variation in UCI move notation.
    pub pv: &'a [&'a str],
    /// Free text; must be last because the rest of the line is the string.
    pub string: Option<&'a str>,
}

impl Info<'_> {
    /// The line without its trailing newline, or `None` when there is nothing to say.
    pub fn render(&self) -> Option<String> {
        let mut s = String::from("info");
        let mut any = false;
        let mut field = |s: &mut String, name: &str, value: fmt::Arguments<'_>| {
            any = true;
            let _ = write!(s, " {name} {value}");
        };
        if let Some(v) = self.depth {
            field(&mut s, "depth", format_args!("{v}"));
        }
        if let Some(v) = self.seldepth {
            field(&mut s, "seldepth", format_args!("{v}"));
        }
        if let Some(v) = self.multipv {
            field(&mut s, "multipv", format_args!("{v}"));
        }
        if let Some(score) = self.score {
            match score {
                Score::Cp(cp) => field(&mut s, "score cp", format_args!("{cp}")),
                Score::Mate(moves) => field(&mut s, "score mate", format_args!("{moves}")),
            }
            match self.bound {
                Bound::Exact => {}
                Bound::Lower => s.push_str(" lowerbound"),
                Bound::Upper => s.push_str(" upperbound"),
            }
        }
        if let Some((w, d, l)) = self.wdl {
            field(&mut s, "wdl", format_args!("{w} {d} {l}"));
        }
        if let Some(v) = self.time {
            field(&mut s, "time", format_args!("{v}"));
        }
        if let Some(v) = self.nodes {
            field(&mut s, "nodes", format_args!("{v}"));
        }
        if let Some(v) = self.nps {
            field(&mut s, "nps", format_args!("{v}"));
        }
        if let Some(v) = self.hashfull {
            field(&mut s, "hashfull", format_args!("{v}"));
        }
        if let Some(v) = self.tbhits {
            field(&mut s, "tbhits", format_args!("{v}"));
        }
        if let Some(v) = self.sbhits {
            field(&mut s, "sbhits", format_args!("{v}"));
        }
        if let Some(v) = self.cpuload {
            field(&mut s, "cpuload", format_args!("{v}"));
        }
        if let Some(v) = self.currmove {
            field(&mut s, "currmove", format_args!("{v}"));
        }
        if let Some(v) = self.currmovenumber {
            field(&mut s, "currmovenumber", format_args!("{v}"));
        }
        if !self.refutation.is_empty() {
            field(
                &mut s,
                "refutation",
                format_args!("{}", self.refutation.join(" ")),
            );
        }
        if let Some((cpu, moves)) = self.currline {
            match cpu {
                Some(cpu) => field(
                    &mut s,
                    "currline",
                    format_args!("{cpu} {}", moves.join(" ")),
                ),
                None => field(&mut s, "currline", format_args!("{}", moves.join(" "))),
            }
        }
        if !self.pv.is_empty() {
            field(&mut s, "pv", format_args!("{}", self.pv.join(" ")));
        }
        if let Some(text) = self.string {
            field(&mut s, "string", format_args!("{text}"));
        }
        any.then_some(s)
    }
}

/// The value part of an `option` declaration; one variant per UCI option type.
#[derive(Debug, Clone, PartialEq)]
pub enum OptionType<'a> {
    /// `type check default true|false`.
    Check { default: bool },
    /// `type spin default <d> min <lo> max <hi>`.
    Spin { default: i64, min: i64, max: i64 },
    /// `type combo default <d> var <a> var <b> ...`; `default` must be one of `vars`.
    Combo {
        default: &'a str,
        vars: &'a [&'a str],
    },
    /// `type button`.
    Button,
    /// `type string default <d>`; an empty default is written as `<empty>`.
    String { default: &'a str },
}

/// One `option name <id> type ...` declaration, sent after `id` and before `uciok`.
#[derive(Debug, Clone, PartialEq)]
pub struct OptionSpec<'a> {
    pub name: &'a str,
    pub kind: OptionType<'a>,
}

impl OptionSpec<'_> {
    /// The line without its trailing newline.
    pub fn render(&self) -> String {
        let mut s = format!("option name {} type ", self.name);
        match &self.kind {
            OptionType::Check { default } => {
                let _ = write!(s, "check default {default}");
            }
            OptionType::Spin { default, min, max } => {
                let _ = write!(s, "spin default {default} min {min} max {max}");
            }
            OptionType::Combo { default, vars } => {
                let _ = write!(s, "combo default {default}");
                for var in vars.iter() {
                    let _ = write!(s, " var {var}");
                }
            }
            OptionType::Button => s.push_str("button"),
            OptionType::String { default } => {
                let shown = if default.is_empty() {
                    "<empty>"
                } else {
                    default
                };
                let _ = write!(s, "string default {shown}");
            }
        }
        s
    }
}

/// Write one `option` declaration.
pub fn option(out: &mut impl Write, spec: &OptionSpec<'_>) -> io::Result<()> {
    line(out, &spec.render())
}

fn line(out: &mut impl Write, text: &str) -> io::Result<()> {
    writeln!(out, "{text}")?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(write: impl FnOnce(&mut Vec<u8>) -> io::Result<()>) -> String {
        let mut buffer = Vec::new();
        write(&mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }

    #[test]
    fn handshake_lines() {
        assert_eq!(
            capture(|o| id(o, "Engine 0.1.0", "engine authors")),
            "id name Engine 0.1.0\nid author engine authors\n"
        );
        assert_eq!(capture(uciok), "uciok\n");
        assert_eq!(capture(readyok), "readyok\n");
    }

    #[test]
    fn bestmove_with_and_without_ponder() {
        assert_eq!(capture(|o| bestmove(o, "e2e4", None)), "bestmove e2e4\n");
        assert_eq!(
            capture(|o| bestmove(o, "e7e8q", Some("d1d8"))),
            "bestmove e7e8q ponder d1d8\n"
        );
    }

    #[test]
    fn info_fields_in_spec_order() {
        let info = Info {
            depth: Some(12),
            seldepth: Some(18),
            multipv: Some(1),
            score: Some(Score::Cp(31)),
            wdl: Some((412, 501, 87)),
            time: Some(1994),
            nodes: Some(3190),
            nps: Some(1600),
            hashfull: Some(12),
            tbhits: Some(3),
            pv: &["e2e4", "e7e5", "g1f3"],
            ..Info::default()
        };
        assert_eq!(
            info.render().unwrap(),
            "info depth 12 seldepth 18 multipv 1 score cp 31 wdl 412 501 87 time 1994 \
             nodes 3190 nps 1600 hashfull 12 tbhits 3 pv e2e4 e7e5 g1f3"
        );
    }

    #[test]
    fn info_mate_bounds_currmove_and_string_last() {
        let info = Info {
            depth: Some(3),
            score: Some(Score::Mate(-2)),
            bound: Bound::Upper,
            currmove: Some("d1h5"),
            currmovenumber: Some(1),
            refutation: &["d1h5", "g6h5"],
            currline: Some((Some(2), &["e2e4", "c7c5"])),
            string: Some("tablebase hit KRvK"),
            ..Info::default()
        };
        assert_eq!(
            info.render().unwrap(),
            "info depth 3 score mate -2 upperbound currmove d1h5 currmovenumber 1 \
             refutation d1h5 g6h5 currline 2 e2e4 c7c5 string tablebase hit KRvK"
        );
        let lower = Info {
            score: Some(Score::Cp(-15)),
            bound: Bound::Lower,
            ..Info::default()
        };
        assert_eq!(lower.render().unwrap(), "info score cp -15 lowerbound");
        assert_eq!(
            Info {
                nodes: Some(1),
                ..Info::default()
            }
            .render()
            .unwrap(),
            "info nodes 1"
        );
    }

    #[test]
    fn empty_info_writes_nothing() {
        assert_eq!(Info::default().render(), None);
        assert_eq!(
            capture(|o| info_string(o, "weights not loaded")),
            "info string weights not loaded\n"
        );
    }

    #[test]
    fn option_declarations_match_spec_examples() {
        let cases: [(OptionSpec<'_>, &str); 6] = [
            (
                OptionSpec {
                    name: "Nullmove",
                    kind: OptionType::Check { default: true },
                },
                "option name Nullmove type check default true",
            ),
            (
                OptionSpec {
                    name: "Selectivity",
                    kind: OptionType::Spin {
                        default: 2,
                        min: 0,
                        max: 4,
                    },
                },
                "option name Selectivity type spin default 2 min 0 max 4",
            ),
            (
                OptionSpec {
                    name: "Style",
                    kind: OptionType::Combo {
                        default: "Normal",
                        vars: &["Solid", "Normal", "Risky"],
                    },
                },
                "option name Style type combo default Normal var Solid var Normal var Risky",
            ),
            (
                OptionSpec {
                    name: "NalimovPath",
                    kind: OptionType::String { default: "c:\\" },
                },
                "option name NalimovPath type string default c:\\",
            ),
            (
                OptionSpec {
                    name: "Clear Hash",
                    kind: OptionType::Button,
                },
                "option name Clear Hash type button",
            ),
            (
                OptionSpec {
                    name: "WeightsFile",
                    kind: OptionType::String { default: "" },
                },
                "option name WeightsFile type string default <empty>",
            ),
        ];
        for (spec, expected) in cases {
            assert_eq!(spec.render(), expected);
            assert_eq!(capture(|o| option(o, &spec)), format!("{expected}\n"));
        }
    }
}
