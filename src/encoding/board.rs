//! Canonical board tokens: pieces, castling bits, en passant, and the optional context
//! (halfmove clock, repetition count, recent moves) that the `state` and `history` model
//! variants consume.

use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Move, Position, Role, Square};

use crate::position::Game;

/// Piece token vocabulary: 0 empty, 1-6 own, 7-12 opponent.
pub const PIECE_VOCAB: usize = 13;
/// En-passant vocabulary: 0 none, 1-8 canonical file + 1.
pub const EN_PASSANT_VOCAB: usize = 9;
/// Castling bits: own kingside, own queenside, opp kingside, opp queenside.
pub const CASTLING_BITS: u8 = 4;
/// Plies of move history the `history` variant sees, newest first.
pub const HISTORY_PLIES: usize = 8;
/// Components of an encoded move: from, to, promotion.
pub const MOVE_COMPONENTS: usize = 3;
/// Square value meaning "no move" in an encoded move.
pub const NO_MOVE_SQUARE: u8 = 64;
/// Largest repetition count the model distinguishes (fivefold ends the game anyway).
pub const MAX_REPETITION: u8 = 5;

const OWN_KINGSIDE: u8 = 1 << 0;
const OWN_QUEENSIDE: u8 = 1 << 1;
const OPP_KINGSIDE: u8 = 1 << 2;
const OPP_QUEENSIDE: u8 = 1 << 3;

/// `(from, to, promotion)` in the canonical frame; `(64, 64, 0)` is "no move".
pub type EncodedMove = [u8; MOVE_COMPONENTS];

/// Everything the network reads for one position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedBoard {
    /// Piece token per canonical square, a1 = 0 ... h8 = 63.
    pub pieces: [u8; 64],
    /// Castling rights relative to the side to move.
    pub castling: u8,
    /// 0 = none, otherwise canonical file + 1 of the en-passant target square. The target
    /// exists after every double pawn push, whether or not a capture is legal.
    pub en_passant: u8,
    /// Plies since the last capture or pawn move.
    pub halfmove_clock: u16,
    /// Occurrences of this position in the game, including now, in `1..=5`.
    pub repetition_count: u8,
    /// The most recent `HISTORY_PLIES` moves, newest first, padded with "no move".
    pub move_history: [EncodedMove; HISTORY_PLIES],
}

impl EncodedBoard {
    /// The newest move, the same as `move_history[0]`.
    pub fn last_move(&self) -> EncodedMove {
        self.move_history[0]
    }
}

/// Map an absolute square into the side-to-move frame.
#[inline]
pub fn canonical_square(square: Square, turn: Color) -> u8 {
    let index = u32::from(square) as u8;
    match turn {
        Color::White => index,
        Color::Black => 63 - index,
    }
}

/// Invert [`canonical_square`].
#[inline]
pub fn absolute_square(canonical: u8, turn: Color) -> Square {
    let index = match turn {
        Color::White => canonical,
        Color::Black => 63 - canonical,
    };
    Square::new(u32::from(index))
}

fn piece_token(role: Role, own: bool) -> u8 {
    let base = match role {
        Role::Pawn => 1,
        Role::Knight => 2,
        Role::Bishop => 3,
        Role::Rook => 4,
        Role::Queen => 5,
        Role::King => 6,
    };
    if own { base } else { base + 6 }
}

fn promotion_token(role: Option<Role>) -> u8 {
    match role {
        None => 0,
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(Role::Queen) => 4,
        Some(Role::Pawn) | Some(Role::King) => 0,
    }
}

/// The square the king lands on for a castling move, in the absolute frame: the g- or
/// c-file of the king's rank (the same in Chess960).
pub fn castle_king_destination(mv: Move) -> Option<Square> {
    match mv {
        Move::Castle { king, rook } => {
            let side = CastlingSide::from_king_side(king < rook);
            Some(side.king_to(if king.rank() == shakmaty::Rank::First {
                Color::White
            } else {
                Color::Black
            }))
        }
        _ => None,
    }
}

/// `(from, to)` of a move as the encoder sees it: castling is the king's move to its
/// destination square, everything else is the move's own squares.
pub fn move_endpoints(mv: Move) -> (Square, Square) {
    match mv {
        Move::Normal { from, to, .. } | Move::EnPassant { from, to } => (from, to),
        Move::Castle { king, .. } => (
            king,
            castle_king_destination(mv).expect("castle move has a destination"),
        ),
        Move::Put { to, .. } => (to, to),
    }
}

/// Encode a move relative to `turn` (the current side to move, not necessarily the mover).
pub fn encode_context_move(mv: Move, turn: Color) -> EncodedMove {
    let (from, to) = move_endpoints(mv);
    [
        canonical_square(from, turn),
        canonical_square(to, turn),
        promotion_token(mv.promotion()),
    ]
}

const NO_MOVE: EncodedMove = [NO_MOVE_SQUARE, NO_MOVE_SQUARE, 0];

/// Encode the current position of a game.
pub fn encode_board(game: &Game) -> EncodedBoard {
    let position: &Chess = game.position();
    let turn = position.turn();
    let board = position.board();

    let mut pieces = [0u8; 64];
    for (square, piece) in board.clone() {
        pieces[canonical_square(square, turn) as usize] =
            piece_token(piece.role, piece.color == turn);
    }

    let castles = position.castles();
    let mut castling = 0u8;
    if castles.has(turn, CastlingSide::KingSide) {
        castling |= OWN_KINGSIDE;
    }
    if castles.has(turn, CastlingSide::QueenSide) {
        castling |= OWN_QUEENSIDE;
    }
    if castles.has(!turn, CastlingSide::KingSide) {
        castling |= OPP_KINGSIDE;
    }
    if castles.has(!turn, CastlingSide::QueenSide) {
        castling |= OPP_QUEENSIDE;
    }

    let en_passant = position
        .ep_square(EnPassantMode::Always)
        .map_or(0, |square| {
            u8::from(absolute_to_canonical_file(square, turn)) + 1
        });

    let mut move_history = [NO_MOVE; HISTORY_PLIES];
    for (slot, &mv) in move_history
        .iter_mut()
        .zip(game.moves().iter().rev().take(HISTORY_PLIES))
    {
        *slot = encode_context_move(mv, turn);
    }

    EncodedBoard {
        pieces,
        castling,
        en_passant,
        halfmove_clock: u16::try_from(position.halfmoves()).unwrap_or(u16::MAX),
        repetition_count: u8::try_from(game.repetitions())
            .unwrap_or(MAX_REPETITION)
            .clamp(1, MAX_REPETITION),
        move_history,
    }
}

fn absolute_to_canonical_file(square: Square, turn: Color) -> shakmaty::File {
    absolute_square(canonical_square(square, turn), Color::White).file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::CastlingMode;

    fn game(fen: Option<&str>, moves: &str) -> Game {
        let moves: Vec<String> = moves.split_whitespace().map(str::to_string).collect();
        Game::from_uci(fen, &moves, CastlingMode::Standard).unwrap()
    }

    #[test]
    fn canonical_square_rotates_for_black() {
        assert_eq!(canonical_square(Square::A1, Color::White), 0);
        assert_eq!(canonical_square(Square::H8, Color::White), 63);
        assert_eq!(canonical_square(Square::A1, Color::Black), 63);
        assert_eq!(canonical_square(Square::E2, Color::Black), 63 - 12);
        for index in 0..64u8 {
            for turn in [Color::White, Color::Black] {
                let square = absolute_square(index, turn);
                assert_eq!(canonical_square(square, turn), index);
            }
        }
    }

    #[test]
    fn startpos_tokens_from_both_sides() {
        let white = encode_board(&game(None, ""));
        // Own back rank: R N B Q K B N R = 4 2 3 5 6 3 2 4; own pawns on rank 2.
        assert_eq!(&white.pieces[0..8], &[4, 2, 3, 5, 6, 3, 2, 4]);
        assert_eq!(&white.pieces[8..16], &[1; 8]);
        assert_eq!(&white.pieces[48..56], &[7; 8]);
        assert_eq!(&white.pieces[56..64], &[10, 8, 9, 11, 12, 9, 8, 10]);
        assert_eq!(white.castling, 0b1111);
        assert_eq!(white.en_passant, 0);
        assert_eq!(white.halfmove_clock, 0);
        assert_eq!(white.repetition_count, 1);
        assert_eq!(white.move_history, [NO_MOVE; HISTORY_PLIES]);

        // After 1.e4 Black is to move: the frame rotates, so Black's pieces are "own"
        // on ranks 1-2 and White's pawn on e4 appears as an opponent pawn on d5.
        let black = encode_board(&game(None, "e2e4"));
        assert_eq!(&black.pieces[0..8], &[4, 2, 3, 6, 5, 3, 2, 4]);
        assert_eq!(
            black.pieces[canonical_square(Square::E4, Color::Black) as usize],
            7
        );
        assert_eq!(
            canonical_square(Square::E4, Color::Black),
            u32::from(Square::D5) as u8
        );
        assert_eq!(black.en_passant, u8::from(shakmaty::File::D) + 1);
        assert_eq!(black.last_move(), [63 - 12, 63 - 28, 0]);
    }

    #[test]
    fn en_passant_is_set_after_any_double_push() {
        // 1.e4 e5: no capture possible, but the target square exists for White's turn.
        let g = game(None, "e2e4 e7e5");
        assert_eq!(encode_board(&g).en_passant, u8::from(shakmaty::File::E) + 1);
        // From a FEN without the field it is absent.
        let g = game(
            Some("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
            "",
        );
        assert_eq!(encode_board(&g).en_passant, 0);
    }

    #[test]
    fn castling_bits_are_relative_to_the_mover() {
        let g = game(Some("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1"), "");
        // Black to move: own = Black (queenside only), opp = White (kingside only).
        assert_eq!(encode_board(&g).castling, OWN_QUEENSIDE | OPP_KINGSIDE);
    }

    #[test]
    fn context_moves_use_the_king_destination_for_castling_and_promotion_tokens() {
        let g = game(Some("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "e1g1");
        // Black to move now; White's castling e1->g1 rotated: e1 -> d8 (59), g1 -> b8 (57).
        assert_eq!(encode_board(&g).last_move(), [59, 57, 0]);
        let g = game(Some("8/1P2k3/8/8/8/8/4K1p1/8 w - - 0 1"), "b7b8n");
        assert_eq!(encode_board(&g).last_move()[2], 1);
        let g = game(Some("8/1P2k3/8/8/8/8/4K1p1/8 b - - 0 1"), "g2g1q");
        assert_eq!(encode_board(&g).last_move()[2], 4);
    }

    #[test]
    fn history_is_newest_first_and_padded() {
        let g = game(None, "e2e4 e7e5 g1f3");
        let encoded = encode_board(&g);
        assert_eq!(
            encoded.move_history[0],
            encode_context_move(g.moves()[2], Color::Black)
        );
        assert_eq!(
            encoded.move_history[2],
            encode_context_move(g.moves()[0], Color::Black)
        );
        assert_eq!(encoded.move_history[3], NO_MOVE);
        assert_eq!(encoded.repetition_count, 1);
        let repeated = game(None, "g1f3 g8f6 f3g1 f6g8");
        assert_eq!(encode_board(&repeated).repetition_count, 2);
    }
}
