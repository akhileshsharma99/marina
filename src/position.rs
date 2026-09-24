//! The game as the GUI describes it: a start position plus the moves played.
//!
//! UCI resends the whole game before every move (`position startpos moves e2e4 e7e5 ...`),
//! so a [`Game`] is rebuilt from scratch each time rather than updated incrementally. It
//! keeps the current [`Chess`] position and the Zobrist hash of every position reached, which
//! is what threefold-repetition detection needs and what `shakmaty::Chess` alone does not
//! hold.
//!
//! Castling is written as the king's two-square move (`e1g1`), standard chess; Chess960 is
//! not offered because the network's move encoding has no action for a castle whose king
//! stays on its file. The mode is a property of the game so that
//! every parse and print in it agrees.

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position};
use thiserror::Error;

/// Why a `position` command could not be applied. The previous game is kept.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PositionError {
    #[error("invalid FEN {fen:?}: {reason}")]
    InvalidFen { fen: String, reason: String },
    #[error("illegal or malformed move {mv:?} at ply {ply}")]
    IllegalMove { mv: String, ply: usize },
}

/// Which repeated-position and halfmove-clock conditions count as a draw.
///
/// The laws of chess let a player claim a draw at the third repetition and after fifty
/// moves (100 plies) without a capture or pawn move ([`DrawRules::FIDE`]). Inside a search
/// tree engines conventionally treat the *second* occurrence of a position since the root
/// as a draw ([`DrawRules::SEARCH`]): it is cheaper and reaches the same value, since the
/// side that repeated once can repeat again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawRules {
    /// Occurrences of the same position, including the current one, that end the game.
    pub repetitions: usize,
    /// Halfmove clock (plies since the last capture or pawn move) that ends the game.
    pub halfmove_clock: u32,
}

impl DrawRules {
    /// Threefold repetition and the fifty-move rule, as a GUI adjudicates them.
    pub const FIDE: DrawRules = DrawRules {
        repetitions: 3,
        halfmove_clock: 100,
    };
    /// Two-fold repetition and the fifty-move rule, for nodes inside a search tree.
    pub const SEARCH: DrawRules = DrawRules {
        repetitions: 2,
        halfmove_clock: 100,
    };
}

impl Default for DrawRules {
    fn default() -> Self {
        Self::FIDE
    }
}

/// How a finished game ended, for the search's terminal handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameEnd {
    Checkmate {
        winner: Color,
    },
    Stalemate,
    InsufficientMaterial,
    /// The position occurred `DrawRules::repetitions` times.
    Repetition,
    /// The halfmove clock reached `DrawRules::halfmove_clock`.
    HalfmoveClock,
}

/// The current position together with the history that led to it.
#[derive(Debug, PartialEq, Eq)]
pub struct Game {
    castling_mode: CastlingMode,
    position: Chess,
    /// Zobrist hashes of every position in the game, oldest first, including the current one.
    hashes: Vec<Zobrist64>,
    /// Moves played from the start position, in order.
    moves: Vec<Move>,
}

impl Clone for Game {
    fn clone(&self) -> Self {
        Self {
            castling_mode: self.castling_mode,
            position: self.position.clone(),
            hashes: self.hashes.clone(),
            moves: self.moves.clone(),
        }
    }

    /// Reuses this game's buffers: the search resets a scratch game to the root once per
    /// descent, and the derived `clone_from` would allocate two vectors each time.
    fn clone_from(&mut self, source: &Self) {
        self.castling_mode = source.castling_mode;
        self.position.clone_from(&source.position);
        self.hashes.clone_from(&source.hashes);
        self.moves.clone_from(&source.moves);
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new(CastlingMode::Standard)
    }
}

/// Analysis positions often break the strict rules (impossible checks, too much material,
/// stale castling rights or en-passant squares); play them anyway when shakmaty can.
#[allow(clippy::result_large_err)] // shakmaty's error type; it is consumed immediately
fn lenient(error: shakmaty::PositionError<Chess>) -> Result<Chess, shakmaty::PositionError<Chess>> {
    error
        .ignore_invalid_castling_rights()
        .or_else(|error| error.ignore_invalid_ep_square())
        .or_else(|error| error.ignore_too_much_material())
        .or_else(|error| error.ignore_impossible_check())
}

impl Game {
    /// The standard start position, with castling read and written in `castling_mode`.
    pub fn new(castling_mode: CastlingMode) -> Self {
        Self::from_position(Chess::default(), castling_mode)
    }

    /// Build a game from `position [startpos | fen <fen>] [moves ...]`, reading and writing
    /// castling in `castling_mode` (the engine plays standard chess).
    pub fn from_uci(
        fen: Option<&str>,
        moves: &[String],
        castling_mode: CastlingMode,
    ) -> Result<Self, PositionError> {
        let start = match fen {
            None => Chess::default(),
            Some(fen) => fen
                .parse::<Fen>()
                .map_err(|error| PositionError::InvalidFen {
                    fen: fen.to_string(),
                    reason: error.to_string(),
                })?
                .into_position(castling_mode)
                .or_else(lenient)
                .map_err(|error| PositionError::InvalidFen {
                    fen: fen.to_string(),
                    reason: error.to_string(),
                })?,
        };
        let mut game = Self::from_position(start, castling_mode);
        for (ply, mv) in moves.iter().enumerate() {
            game.play_uci(mv).map_err(|_| PositionError::IllegalMove {
                mv: mv.clone(),
                ply,
            })?;
        }
        Ok(game)
    }

    fn from_position(position: Chess, castling_mode: CastlingMode) -> Self {
        let hash = hash_of(&position);
        Self {
            castling_mode,
            position,
            hashes: vec![hash],
            moves: Vec::new(),
        }
    }

    /// Play one move given in UCI notation (`e2e4`, `e7e8q`, `e1g1`).
    pub fn play_uci(&mut self, uci: &str) -> Result<(), PositionError> {
        let mv = UciMove::from_ascii(uci.as_bytes())
            .ok()
            .and_then(|parsed| parsed.to_move(&self.position).ok())
            .ok_or_else(|| PositionError::IllegalMove {
                mv: uci.to_string(),
                ply: self.moves.len(),
            })?;
        self.play(mv);
        Ok(())
    }

    /// Play a move known to be legal in the current position.
    pub fn play(&mut self, mv: Move) {
        // Incremental hash update where shakmaty supports it (no en passant involved),
        // a full recompute otherwise.
        let incremental = self
            .position
            .update_zobrist_hash(self.hash(), mv, EnPassantMode::Legal);
        self.position.play_unchecked(mv);
        let hash = incremental.unwrap_or_else(|| hash_of(&self.position));
        debug_assert_eq!(hash, hash_of(&self.position));
        self.hashes.push(hash);
        self.moves.push(mv);
    }

    /// Take back the last move, restoring `previous`, the position saved before it was
    /// played (`Chess` has no undo of its own; the search saves a copy per ply).
    pub fn undo(&mut self, previous: Chess) {
        debug_assert!(!self.moves.is_empty(), "undo with no move played");
        self.moves.pop();
        self.hashes.pop();
        self.position = previous;
        debug_assert_eq!(self.hash(), hash_of(&self.position));
    }

    pub fn position(&self) -> &Chess {
        &self.position
    }

    pub fn castling_mode(&self) -> CastlingMode {
        self.castling_mode
    }

    pub fn turn(&self) -> Color {
        self.position.turn()
    }

    /// Number of plies played since the start position.
    /// Plies played in the game so far, counting from the game's first move rather than
    /// from the `position` FEN: what a moves-left model needs.
    pub fn game_ply(&self) -> u32 {
        // FEN allows a fullmove counter up to u32::MAX; the ply must not overflow on it.
        let fullmoves = u32::from(self.position.fullmoves()).max(1);
        (fullmoves - 1)
            .saturating_mul(2)
            .saturating_add(u32::from(self.position.turn() == Color::Black))
    }

    pub fn ply(&self) -> usize {
        self.moves.len()
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// The moves that take `earlier` to this game, when this game continues it: same start
    /// position (and castling mode) and `earlier`'s moves are a prefix of this game's.
    /// `Some(&[])` when the two are the same game.
    pub fn continuation_from<'a>(&'a self, earlier: &Game) -> Option<&'a [Move]> {
        if self.castling_mode != earlier.castling_mode
            || self.hashes.len() < earlier.hashes.len()
            || self.hashes[..earlier.hashes.len()] != earlier.hashes[..]
            || self.moves[..earlier.moves.len()] != earlier.moves[..]
        {
            return None;
        }
        Some(&self.moves[earlier.moves.len()..])
    }

    /// Zobrist hash of the current position.
    pub fn hash(&self) -> Zobrist64 {
        *self
            .hashes
            .last()
            .expect("a game always has its current position")
    }

    /// Legal moves in the current position (empty when the game is over).
    pub fn legal_moves(&self) -> shakmaty::MoveList {
        self.position.legal_moves()
    }

    /// UCI notation for a move in this game's castling mode (`e1g1` standard, `e1h1` in
    /// Chess960; promotions as `e7e8q`).
    pub fn uci(&self, mv: Move) -> String {
        mv.to_uci(self.castling_mode).to_string()
    }

    /// How many times the current position has occurred in the game, including now.
    ///
    /// Only positions since the last irreversible move (a capture or a pawn move, which reset
    /// the halfmove clock) can be the current one, and only those with the same side to move,
    /// so the scan is over at most `halfmoves / 2` hashes however long the game is.
    pub fn repetitions(&self) -> usize {
        let current = self.hash();
        let reversible = (self.position.halfmoves() as usize).min(self.hashes.len() - 1);
        let earliest = self.hashes.len() - 1 - reversible;
        1 + self.hashes[earliest..self.hashes.len() - 1]
            .iter()
            .rev()
            .skip(1)
            .step_by(2)
            .filter(|&&hash| hash == current)
            .count()
    }

    /// Whether the game is over under the laws of chess ([`DrawRules::FIDE`]), and how.
    pub fn game_end(&self) -> Option<GameEnd> {
        self.game_end_with(DrawRules::FIDE)
    }

    /// Whether the game is over under the given draw rules, and how.
    pub fn game_end_with(&self, rules: DrawRules) -> Option<GameEnd> {
        self.game_end_given(rules, &self.legal_moves())
    }

    /// [`Self::game_end_with`] for a caller that has already generated the legal moves,
    /// so mate and stalemate detection do not generate them again.
    pub fn game_end_given(&self, rules: DrawRules, legal: &shakmaty::MoveList) -> Option<GameEnd> {
        if legal.is_empty() {
            return Some(if self.position.is_check() {
                GameEnd::Checkmate {
                    winner: !self.position.turn(),
                }
            } else {
                GameEnd::Stalemate
            });
        }
        if self.position.is_insufficient_material() {
            return Some(GameEnd::InsufficientMaterial);
        }
        if self.repetitions() >= rules.repetitions {
            return Some(GameEnd::Repetition);
        }
        if self.position.halfmoves() >= rules.halfmove_clock {
            return Some(GameEnd::HalfmoveClock);
        }
        None
    }
}

fn hash_of(position: &Chess) -> Zobrist64 {
    position.zobrist_hash(EnPassantMode::Legal)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STD: CastlingMode = CastlingMode::Standard;

    fn moves(list: &str) -> Vec<String> {
        list.split_whitespace().map(str::to_string).collect()
    }

    fn game(fen: Option<&str>, list: &str) -> Game {
        Game::from_uci(fen, &moves(list), STD).unwrap()
    }

    #[test]
    fn startpos_and_moves() {
        let g = game(None, "e2e4 e7e5 g1f3");
        assert_eq!(g.ply(), 3);
        assert_eq!(g.turn(), Color::Black);
        assert_eq!(g.legal_moves().len(), 29);
        assert_eq!(g.game_end(), None);
        let uci: Vec<String> = g.moves().iter().map(|&m| g.uci(m)).collect();
        assert_eq!(uci, vec!["e2e4", "e7e5", "g1f3"]);
        assert_eq!(Game::default(), game(None, ""));
        assert_eq!(
            Game::new(CastlingMode::Chess960).castling_mode(),
            CastlingMode::Chess960
        );
        assert_eq!(
            Game::new(CastlingMode::Chess960).position(),
            Game::default().position()
        );
    }

    #[test]
    fn fen_with_four_or_six_fields_and_castling_notation() {
        let g = game(Some("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "e1g1 e8c8");
        assert_eq!(g.ply(), 2);
        assert_eq!(g.uci(g.moves()[0]), "e1g1");
        assert_eq!(g.uci(g.moves()[1]), "e8c8");
        let short = game(Some("8/8/8/8/8/8/8/K6k w - -"), "");
        assert_eq!(short.legal_moves().len(), 3);
    }

    #[test]
    fn chess960_castling_notation() {
        // Same rooks-and-kings position: castling is king-takes-rook in Chess960 mode.
        let g = Game::from_uci(
            Some("r3k2r/8/8/8/8/8/8/R3K2R w HAha - 0 1"),
            &moves("e1h1 e8a8"),
            CastlingMode::Chess960,
        )
        .unwrap();
        assert_eq!(g.castling_mode(), CastlingMode::Chess960);
        assert_eq!(g.uci(g.moves()[0]), "e1h1");
        assert_eq!(g.uci(g.moves()[1]), "e8a8");
        // A real Chess960 start position (king c1, rooks b1/d1): its castling rights are
        // only meaningful in 960 mode; in standard mode they are dropped, not rejected.
        let fen = "qrkrnbbn/pppppppp/8/8/8/8/PPPPPPPP/QRKRNBBN w KQkq - 0 1";
        let start = Game::from_uci(Some(fen), &[], CastlingMode::Chess960).unwrap();
        assert!(!start.legal_moves().is_empty());
        let standard = Game::from_uci(Some(fen), &[], STD).unwrap();
        assert!(standard.position().castles().is_empty());
    }

    #[test]
    fn promotions_round_trip() {
        let g = game(Some("8/1P2k3/8/8/8/8/4K1p1/8 w - - 0 1"), "b7b8q g2g1n");
        assert_eq!(g.uci(g.moves()[0]), "b7b8q");
        assert_eq!(g.uci(g.moves()[1]), "g2g1n");
    }

    #[test]
    fn errors_name_the_offending_input() {
        let bad_fen = Game::from_uci(Some("not a fen"), &[], STD).unwrap_err();
        assert!(matches!(bad_fen, PositionError::InvalidFen { ref fen, .. } if fen == "not a fen"));
        assert!(
            bad_fen
                .to_string()
                .starts_with("invalid FEN \"not a fen\": ")
        );
        assert_eq!(
            Game::from_uci(None, &moves("e2e4 e7e5 e4e5"), STD).unwrap_err(),
            PositionError::IllegalMove {
                mv: "e4e5".into(),
                ply: 2
            }
        );
        assert_eq!(
            Game::from_uci(None, &moves("zz99"), STD).unwrap_err(),
            PositionError::IllegalMove {
                mv: "zz99".into(),
                ply: 0
            }
        );
    }

    #[test]
    fn game_end_detection() {
        let mate = game(None, "f2f3 e7e5 g2g4 d8h4");
        assert_eq!(
            mate.game_end(),
            Some(GameEnd::Checkmate {
                winner: Color::Black
            })
        );
        assert!(mate.legal_moves().is_empty());

        let stalemate = game(Some("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), "");
        assert_eq!(stalemate.game_end(), Some(GameEnd::Stalemate));
        let mated = game(Some("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"), "");
        assert_eq!(
            mated.game_end(),
            Some(GameEnd::Checkmate {
                winner: Color::White
            })
        );

        let bare = game(Some("8/8/8/8/8/8/8/K6k w - - 0 1"), "");
        assert_eq!(bare.game_end(), Some(GameEnd::InsufficientMaterial));

        let fifty = game(Some("8/8/4k3/8/8/4K3/4R3/8 w - - 100 120"), "");
        assert_eq!(fifty.game_end(), Some(GameEnd::HalfmoveClock));
        let ninety_nine = game(Some("8/8/4k3/8/8/4K3/4R3/8 w - - 99 120"), "");
        assert_eq!(ninety_nine.game_end(), None);

        let shuffle = game(None, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8");
        assert_eq!(shuffle.repetitions(), 3);
        assert_eq!(shuffle.game_end(), Some(GameEnd::Repetition));
        let twice = game(None, "g1f3 g8f6 f3g1 f6g8");
        assert_eq!(twice.repetitions(), 2);
        assert_eq!(twice.game_end(), None);
        assert_eq!(
            twice.game_end_with(DrawRules::SEARCH),
            Some(GameEnd::Repetition)
        );
        assert_eq!(DrawRules::default(), DrawRules::FIDE);
    }

    #[test]
    fn hash_distinguishes_side_to_move_and_castling_rights() {
        let a = game(None, "e2e4");
        let b = game(
            Some("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"),
            "",
        );
        assert_ne!(a.hash(), b.hash());
        let c = game(
            Some("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            "",
        );
        assert_eq!(a.hash(), c.hash());
    }

    #[test]
    fn repetitions_count_only_reversible_history_with_the_same_side_to_move() {
        // Knights out and back: the start position recurs, and so does the position after
        // 1.Nf3 (with Black to move). A pawn move in between makes everything before it
        // unreachable, whatever the hash list holds.
        let g = game(None, "g1f3 g8f6 f3g1 f6g8");
        assert_eq!(g.repetitions(), 2);
        let g = game(None, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8");
        assert_eq!(g.repetitions(), 3);
        // After the pawn moves the start position is unreachable; the position after
        // 3...e5 recurs once (the knights out and back again), so two, not four.
        let g = game(None, "g1f3 g8f6 f3g1 f6g8 e2e4 e7e5 g1f3 g8f6 f3g1 f6g8");
        assert_eq!(g.repetitions(), 2);
        // The old scan would have found the same answer here; what it could not do is stop.
        let g = game(None, "g1f3 g8f6 f3g1 f6g8 e2e4 e7e5");
        assert_eq!(g.repetitions(), 1);
        // A position that matches one with the other side to move is not a repetition.
        let g = game(None, "g1f3 g8f6 f3g1");
        assert_eq!(g.repetitions(), 1);
    }

    #[test]
    fn game_ply_survives_the_largest_fullmove_counter() {
        let g = game(
            Some("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 4294967295"),
            "",
        );
        assert_eq!(g.game_ply(), u32::MAX);
    }
}
