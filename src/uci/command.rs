//! GUI-to-engine commands of the UCI protocol and their parser.
//!
//! Specification: Stefan Meyer-Kahlen, *Description of the universal chess interface (UCI)*,
//! April 2006, <https://backscattering.de/chess/uci/> (canonical text distributed with
//! Shredder, <https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html>).
//!
//! One line of input becomes one [`Command`]. The parser follows the specification's
//! leniency rules: tokens are separated by arbitrary whitespace, an unknown leading token is
//! skipped and parsing continues with the rest of the line (`joho debug on` is `debug on`),
//! unknown tokens inside a command are ignored, and a line with nothing recognisable becomes
//! [`Command::Unknown`] so the caller can ignore it without failing.

use std::str::SplitWhitespace;
use strum::EnumString;

/// A command the GUI can send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// `uci`: identify and list options, then `uciok`.
    Uci,
    /// `debug on|off`.
    Debug(bool),
    /// `isready`: answer `readyok` once idle.
    IsReady,
    /// `setoption name <id> [value <x>]`; `value` is `None` for button options.
    SetOption { name: String, value: Option<String> },
    /// `register ...`: the engine needs no registration, so the payload is not kept.
    Register,
    /// `ucinewgame`: the next search is from a different game.
    UciNewGame,
    /// `position [fen <fen> | startpos] [moves <m1> ... <mn>]`; `fen` is `None` for `startpos`.
    Position {
        fen: Option<String>,
        moves: Vec<String>,
    },
    /// `go` with its limits.
    Go(GoLimits),
    /// `stop`: finish as soon as possible and send `bestmove`.
    Stop,
    /// `ponderhit`: the pondered move was played; continue as a normal search.
    PonderHit,
    /// `quit`.
    Quit,
    /// Nothing recognisable; the payload is the original line.
    Unknown(String),
}

/// Constraints on one search; every field is optional and absent fields must not
/// influence the search.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GoLimits {
    /// Restrict the root to these moves (UCI notation).
    pub searchmoves: Option<Vec<String>>,
    /// Pondering mode: do not stop on your own, even at mate.
    pub ponder: bool,
    /// Milliseconds left on White's clock.
    pub wtime: Option<u64>,
    /// Milliseconds left on Black's clock.
    pub btime: Option<u64>,
    /// White's increment per move in milliseconds.
    pub winc: Option<u64>,
    /// Black's increment per move in milliseconds.
    pub binc: Option<u64>,
    /// Moves until the next time control; absent means sudden death.
    pub movestogo: Option<u32>,
    /// Stop once the mean simulation depth reaches this many plies.
    pub depth: Option<u32>,
    /// Search this many nodes only.
    pub nodes: Option<u64>,
    /// Search for a mate in this many moves: the search runs until a mate is proven or
    /// `stop` (the distance is not enforced).
    pub mate: Option<u32>,
    /// Search exactly this many milliseconds.
    pub movetime: Option<u64>,
    /// Search until `stop`.
    pub infinite: bool,
}

impl GoLimits {
    /// True when no limit at all was given (`go` alone), which the spec treats like `infinite`.
    pub fn is_unbounded(&self) -> bool {
        self.wtime.is_none()
            && self.btime.is_none()
            && self.movetime.is_none()
            && self.nodes.is_none()
            && self.depth.is_none()
            && self.mate.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString)]
#[strum(serialize_all = "lowercase")]
enum Keyword {
    Uci,
    Debug,
    IsReady,
    SetOption,
    Register,
    UciNewGame,
    Position,
    Go,
    Stop,
    PonderHit,
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString)]
#[strum(serialize_all = "lowercase")]
enum GoKeyword {
    SearchMoves,
    Ponder,
    WTime,
    BTime,
    WInc,
    BInc,
    MovesToGo,
    Depth,
    Nodes,
    Mate,
    MoveTime,
    Infinite,
}

/// Parse one line of GUI input.
pub fn parse(line: &str) -> Command {
    let mut tokens = line.split_whitespace();
    // Skip unknown leading tokens, as the specification asks.
    let keyword = loop {
        match tokens.next() {
            None => return Command::Unknown(line.to_string()),
            Some(token) => {
                if let Ok(keyword) = token.parse::<Keyword>() {
                    break keyword;
                }
            }
        }
    };
    match keyword {
        Keyword::Uci => Command::Uci,
        Keyword::Debug => parse_debug(tokens),
        Keyword::IsReady => Command::IsReady,
        Keyword::SetOption => parse_setoption(tokens),
        Keyword::Register => Command::Register,
        Keyword::UciNewGame => Command::UciNewGame,
        Keyword::Position => parse_position(tokens),
        Keyword::Go => Command::Go(parse_go(tokens)),
        Keyword::Stop => Command::Stop,
        Keyword::PonderHit => Command::PonderHit,
        Keyword::Quit => Command::Quit,
    }
}

fn parse_debug(mut tokens: SplitWhitespace<'_>) -> Command {
    match tokens.next() {
        Some("off") => Command::Debug(false),
        // `debug on`; a missing or unknown argument is treated as `on`.
        _ => Command::Debug(true),
    }
}

/// `setoption name <id> [value <x>]`: the name runs from `name` to `value` (or the end),
/// the value from `value` to the end. Both may contain spaces.
fn parse_setoption(tokens: SplitWhitespace<'_>) -> Command {
    let mut name: Vec<&str> = Vec::new();
    let mut value: Option<Vec<&str>> = None;
    let mut in_name = false;
    for token in tokens {
        match token {
            "name" if value.is_none() && name.is_empty() => in_name = true,
            "value" if value.is_none() => {
                in_name = false;
                value = Some(Vec::new());
            }
            _ => {
                if let Some(value) = value.as_mut() {
                    value.push(token);
                } else if in_name {
                    name.push(token);
                }
            }
        }
    }
    if name.is_empty() {
        return Command::Unknown(String::from("setoption"));
    }
    Command::SetOption {
        name: name.join(" "),
        value: value.map(|words| words.join(" ")),
    }
}

/// `position [fen <fields...> | startpos] [moves <m1> ... <mn>]`. FEN fields run until
/// `moves` or the end of the line, so four- to six-field FENs both work.
fn parse_position(mut tokens: SplitWhitespace<'_>) -> Command {
    let mut fen: Option<Vec<&str>> = None;
    loop {
        match tokens.next() {
            None => break,
            Some("startpos") => {
                fen = None;
                break;
            }
            Some("fen") => {
                fen = Some(Vec::new());
                break;
            }
            Some(_) => continue, // unknown token: skip
        }
    }
    let mut moves: Vec<String> = Vec::new();
    let mut in_moves = false;
    for token in tokens {
        if token == "moves" && !in_moves {
            in_moves = true;
        } else if in_moves {
            moves.push(token.to_string());
        } else if let Some(fields) = fen.as_mut() {
            fields.push(token);
        }
    }
    Command::Position {
        fen: fen.map(|fields| fields.join(" ")),
        moves,
    }
}

/// `go` sub-commands in any order; a value that fails to parse leaves its field untouched,
/// and `searchmoves` collects moves until the next keyword.
fn parse_go(tokens: SplitWhitespace<'_>) -> GoLimits {
    let mut limits = GoLimits::default();
    let mut tokens = tokens.peekable();
    while let Some(token) = tokens.next() {
        let Ok(keyword) = token.parse::<GoKeyword>() else {
            continue;
        };
        match keyword {
            GoKeyword::SearchMoves => {
                let mut moves = Vec::new();
                while let Some(next) = tokens.peek() {
                    if next.parse::<GoKeyword>().is_ok() {
                        break;
                    }
                    moves.push(tokens.next().unwrap_or_default().to_string());
                }
                limits.searchmoves = Some(moves);
            }
            GoKeyword::Ponder => limits.ponder = true,
            GoKeyword::Infinite => limits.infinite = true,
            GoKeyword::WTime => limits.wtime = number(tokens.next()),
            GoKeyword::BTime => limits.btime = number(tokens.next()),
            GoKeyword::WInc => limits.winc = number(tokens.next()),
            GoKeyword::BInc => limits.binc = number(tokens.next()),
            GoKeyword::MovesToGo => limits.movestogo = number(tokens.next()),
            GoKeyword::Depth => limits.depth = number(tokens.next()),
            GoKeyword::Nodes => limits.nodes = number(tokens.next()),
            GoKeyword::Mate => limits.mate = number(tokens.next()),
            GoKeyword::MoveTime => limits.movetime = number(tokens.next()),
        }
    }
    limits
}

fn number<T: std::str::FromStr>(token: Option<&str>) -> Option<T> {
    token.and_then(|value| value.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_keywords() {
        assert_eq!(parse("uci"), Command::Uci);
        assert_eq!(parse("isready"), Command::IsReady);
        assert_eq!(parse("ucinewgame"), Command::UciNewGame);
        assert_eq!(parse("stop"), Command::Stop);
        assert_eq!(parse("ponderhit"), Command::PonderHit);
        assert_eq!(parse("quit"), Command::Quit);
        assert_eq!(parse("  isready \t"), Command::IsReady);
    }

    #[test]
    fn debug_defaults_to_on() {
        assert_eq!(parse("debug on"), Command::Debug(true));
        assert_eq!(parse("debug off"), Command::Debug(false));
        assert_eq!(parse("debug"), Command::Debug(true));
    }

    #[test]
    fn unknown_leading_tokens_are_skipped() {
        assert_eq!(parse("joho debug on"), Command::Debug(true));
        assert_eq!(parse("debug joho on"), Command::Debug(true));
        assert_eq!(parse(""), Command::Unknown(String::new()));
        assert_eq!(parse("hello world"), Command::Unknown("hello world".into()));
    }

    #[test]
    fn setoption_with_spaces_and_buttons() {
        assert_eq!(
            parse("setoption name Nullmove value true"),
            Command::SetOption {
                name: "Nullmove".into(),
                value: Some("true".into())
            }
        );
        assert_eq!(
            parse("setoption name Clear Hash"),
            Command::SetOption {
                name: "Clear Hash".into(),
                value: None
            }
        );
        assert_eq!(
            parse("setoption name Move Overhead value 100"),
            Command::SetOption {
                name: "Move Overhead".into(),
                value: Some("100".into())
            }
        );
        assert_eq!(
            parse(r"setoption name NalimovPath value c:\chess\tb\4;c:\chess\tb\5"),
            Command::SetOption {
                name: "NalimovPath".into(),
                value: Some(r"c:\chess\tb\4;c:\chess\tb\5".into())
            }
        );
        assert_eq!(
            parse("setoption name WeightsFile value /path with space/model"),
            Command::SetOption {
                name: "WeightsFile".into(),
                value: Some("/path with space/model".into())
            }
        );
        assert_eq!(parse("setoption"), Command::Unknown("setoption".into()));
    }

    #[test]
    fn register_forms() {
        assert_eq!(parse("register later"), Command::Register);
        assert_eq!(
            parse("register name Stefan MK code 4359874324"),
            Command::Register
        );
    }

    #[test]
    fn position_startpos_and_fen() {
        assert_eq!(
            parse("position startpos"),
            Command::Position {
                fen: None,
                moves: vec![]
            }
        );
        assert_eq!(
            parse("position startpos moves e2e4 e7e5 g1f3"),
            Command::Position {
                fen: None,
                moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into()]
            }
        );
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(
            parse(&format!("position fen {fen} moves e2e4")),
            Command::Position {
                fen: Some(fen.into()),
                moves: vec!["e2e4".into()]
            }
        );
        assert_eq!(
            parse("position fen 8/8/8/8/8/8/8/K6k w - -"),
            Command::Position {
                fen: Some("8/8/8/8/8/8/8/K6k w - -".into()),
                moves: vec![]
            }
        );
        assert_eq!(
            parse("position"),
            Command::Position {
                fen: None,
                moves: vec![]
            }
        );
    }

    #[test]
    fn go_all_limits_in_any_order() {
        let Command::Go(limits) = parse(
            "go ponder wtime 118000 btime 117000 winc 1000 binc 1000 movestogo 40 \
             depth 12 nodes 800 mate 3 movetime 5000 infinite searchmoves e2e4 d2d4",
        ) else {
            panic!("expected go");
        };
        assert_eq!(
            limits,
            GoLimits {
                searchmoves: Some(vec!["e2e4".into(), "d2d4".into()]),
                ponder: true,
                wtime: Some(118_000),
                btime: Some(117_000),
                winc: Some(1000),
                binc: Some(1000),
                movestogo: Some(40),
                depth: Some(12),
                nodes: Some(800),
                mate: Some(3),
                movetime: Some(5000),
                infinite: true,
            }
        );
    }

    #[test]
    fn go_searchmoves_stops_at_next_keyword() {
        let Command::Go(limits) = parse("go searchmoves e2e4 d2d4 wtime 1000") else {
            panic!("expected go");
        };
        assert_eq!(limits.searchmoves, Some(vec!["e2e4".into(), "d2d4".into()]));
        assert_eq!(limits.wtime, Some(1000));
    }

    #[test]
    fn go_alone_is_unbounded_and_bad_numbers_are_ignored() {
        let Command::Go(limits) = parse("go") else {
            panic!("expected go");
        };
        assert_eq!(limits, GoLimits::default());
        assert!(limits.is_unbounded());
        let Command::Go(limits) = parse("go wtime abc nodes 800 bogus 7") else {
            panic!("expected go");
        };
        assert_eq!(limits.wtime, None);
        assert_eq!(limits.nodes, Some(800));
        assert!(!limits.is_unbounded());
        let Command::Go(limits) = parse("go infinite") else {
            panic!("expected go");
        };
        assert!(limits.infinite && limits.is_unbounded());
    }
}
