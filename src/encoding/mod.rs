//! The network's view of a position: canonical board tokens and the 73-plane action space.
//!
//! The networks were trained on exactly this encoding, so it must not drift: `model.json`
//! documents the layout and `vectors.npz` holds reference encodings that the
//! `golden_encoding` test and `verify` check against.
//!
//! Everything is expressed in the **canonical frame**: the side to move is "own", and when
//! Black is to move the board is rotated 180 degrees (square `s` becomes `63 - s`) so own
//! pawns always advance towards rank 8. Own pieces are tokens 1-6, the opponent's 7-12.

pub mod action;
pub mod board;

pub use action::{ACTION_SIZE, LegalActions, PLANES_PER_SQUARE, decode_action, encode_move};
pub use board::{
    CASTLING_BITS, EN_PASSANT_VOCAB, EncodedBoard, EncodedMove, HISTORY_PLIES, MAX_REPETITION,
    MOVE_COMPONENTS, NO_MOVE_SQUARE, PIECE_VOCAB, canonical_square, encode_board,
    encode_context_move,
};
