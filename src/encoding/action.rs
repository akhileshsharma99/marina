//! The 73-plane action space: `action = canonical_from_square * 73 + plane`.
//!
//! Planes, for a move from the canonical origin square with displacement `(dfile, drank)`:
//!
//! * `0..56`  queen rays: `direction * 7 + (distance - 1)`, directions clockwise from north
//!   `(0,1) (1,1) (1,0) (1,-1) (0,-1) (-1,-1) (-1,0) (-1,1)`; queen promotions use these;
//! * `56..64` knight jumps in the order `(1,2) (2,1) (2,-1) (1,-2) (-1,-2) (-2,-1) (-2,1) (-1,2)`;
//! * `64..73` underpromotions: `64 + (dfile + 1) * 3 + piece`, piece 0 knight, 1 bishop, 2 rook.
//!
//! Castling is encoded as the king's move (`e1g1`, a two-square ray), the convention the
//! networks were trained with (the golden vectors check it).

use shakmaty::{Chess, Move, Position, Role};

use super::board::{absolute_square, canonical_square, move_endpoints};
use crate::position::Game;

pub const PLANES_PER_SQUARE: usize = 73;
pub const ACTION_SIZE: usize = 64 * PLANES_PER_SQUARE;

const QUEEN_DIRECTIONS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];
const KNIGHT_OFFSETS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const UNDERPROMOTIONS: [Role; 3] = [Role::Knight, Role::Bishop, Role::Rook];

fn coordinates(canonical: u8) -> (i8, i8) {
    ((canonical % 8) as i8, (canonical / 8) as i8)
}

fn plane(dfile: i8, drank: i8, promotion: Option<Role>) -> Option<usize> {
    if let Some(index) = promotion.and_then(|role| UNDERPROMOTIONS.iter().position(|&r| r == role))
    {
        if drank != 1 || !(-1..=1).contains(&dfile) {
            return None;
        }
        return Some(64 + (dfile + 1) as usize * 3 + index);
    }
    if let Some(index) = KNIGHT_OFFSETS
        .iter()
        .position(|&(f, r)| f == dfile && r == drank)
    {
        return Some(56 + index);
    }
    let distance = dfile.abs().max(drank.abs());
    if distance == 0 || !(dfile == 0 || drank == 0 || dfile.abs() == drank.abs()) {
        return None;
    }
    let direction = (dfile / distance, drank / distance);
    QUEEN_DIRECTIONS
        .iter()
        .position(|&d| d == direction)
        .map(|index| index * 7 + (distance as usize - 1))
}

/// Encode a legal move of `position` as its action index.
pub fn encode_move(position: &Chess, mv: Move) -> usize {
    let turn = position.turn();
    let (from, to) = move_endpoints(mv);
    let canonical_from = canonical_square(from, turn);
    let canonical_to = canonical_square(to, turn);
    let (from_file, from_rank) = coordinates(canonical_from);
    let (to_file, to_rank) = coordinates(canonical_to);
    let plane = plane(to_file - from_file, to_rank - from_rank, mv.promotion())
        .expect("a legal chess move always has an action plane");
    canonical_from as usize * PLANES_PER_SQUARE + plane
}

/// Decode an action index into the legal move of `position` it denotes, if any.
pub fn decode_action(position: &Chess, action: usize) -> Option<Move> {
    if action >= ACTION_SIZE {
        return None;
    }
    let canonical_from = (action / PLANES_PER_SQUARE) as u8;
    let plane = action % PLANES_PER_SQUARE;
    let (dfile, drank, promotion) = decode_plane(plane)?;
    let (from_file, from_rank) = coordinates(canonical_from);
    let to_file = from_file + dfile;
    let to_rank = from_rank + drank;
    if !(0..8).contains(&to_file) || !(0..8).contains(&to_rank) {
        return None;
    }
    let turn = position.turn();
    let from = absolute_square(canonical_from, turn);
    let to = absolute_square((to_rank * 8 + to_file) as u8, turn);
    // A ray move by a pawn onto the last rank is a queen promotion.
    let promotion = promotion.or_else(|| {
        let is_pawn = position
            .board()
            .piece_at(from)
            .is_some_and(|piece| piece.role == Role::Pawn && piece.color == turn);
        (is_pawn && to_rank == 7).then_some(Role::Queen)
    });
    position.legal_moves().into_iter().find(|&candidate| {
        let (candidate_from, candidate_to) = move_endpoints(candidate);
        candidate_from == from && candidate_to == to && candidate.promotion() == promotion
    })
}

fn decode_plane(plane: usize) -> Option<(i8, i8, Option<Role>)> {
    match plane {
        0..56 => {
            let (direction, distance) = (plane / 7, (plane % 7) as i8 + 1);
            let (df, dr) = QUEEN_DIRECTIONS[direction];
            Some((df * distance, dr * distance, None))
        }
        56..64 => {
            let (df, dr) = KNIGHT_OFFSETS[plane - 56];
            Some((df, dr, None))
        }
        64..PLANES_PER_SQUARE => {
            let (direction, piece) = ((plane - 64) / 3, (plane - 64) % 3);
            Some((direction as i8 - 1, 1, Some(UNDERPROMOTIONS[piece])))
        }
        _ => None,
    }
}

/// The legal moves of a position paired with their action indices; [`LegalActions::mask`]
/// derives the legal mask.
#[derive(Debug, Clone)]
pub struct LegalActions {
    pub moves: Vec<(Move, usize)>,
}

impl LegalActions {
    pub fn of(game: &Game) -> Self {
        Self::of_position(game.position())
    }

    pub fn of_position(position: &Chess) -> Self {
        Self::from_moves(position, &position.legal_moves())
    }

    /// Pair already-generated legal moves of `position` with their action indices.
    pub fn from_moves(position: &Chess, legal: &shakmaty::MoveList) -> Self {
        let mut moves = Vec::with_capacity(legal.len());
        moves.extend(legal.iter().map(|&mv| (mv, encode_move(position, mv))));
        Self { moves }
    }

    /// Boolean mask over all `ACTION_SIZE` actions.
    pub fn mask(&self) -> Vec<bool> {
        let mut mask = vec![false; ACTION_SIZE];
        for &(_, action) in &self.moves {
            mask[action] = true;
        }
        mask
    }

    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    pub fn len(&self) -> usize {
        self.moves.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::CastlingMode;

    fn game(fen: Option<&str>, moves: &str) -> Game {
        let moves: Vec<String> = moves.split_whitespace().map(str::to_string).collect();
        Game::from_uci(fen, &moves, CastlingMode::Standard).unwrap()
    }

    fn action_of(g: &Game, uci: &str) -> usize {
        let mv = shakmaty::uci::UciMove::from_ascii(uci.as_bytes())
            .unwrap()
            .to_move(g.position())
            .unwrap();
        encode_move(g.position(), mv)
    }

    #[test]
    fn hand_checked_actions() {
        let start = game(None, "");
        // e2e4: from e2 (12), north two squares: direction 0, distance 2 -> plane 1.
        assert_eq!(action_of(&start, "e2e4"), 12 * 73 + 1);
        // g1f3: knight (-1, 2) is knight index 7 -> plane 63.
        assert_eq!(action_of(&start, "g1f3"), 6 * 73 + 63);
        // Black to move after 1.e4: e7e5 in the rotated frame is from 63-52=11 (d2), north two.
        let black = game(None, "e2e4");
        assert_eq!(action_of(&black, "e7e5"), 11 * 73 + 1);
        // Castling is the king's two-square move east: direction 2 (1,0), distance 2 -> plane 15.
        let castle = game(Some("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "");
        assert_eq!(action_of(&castle, "e1g1"), 4 * 73 + 15);
        // Queen promotion uses the ray plane; underpromotions use 64..73.
        let promo = game(Some("8/1P2k3/8/8/8/8/4K1p1/8 w - - 0 1"), "");
        assert_eq!(action_of(&promo, "b7b8q"), 49 * 73);
        assert_eq!(action_of(&promo, "b7b8n"), 49 * 73 + 64 + 3);
        assert_eq!(action_of(&promo, "b7b8r"), 49 * 73 + 64 + 5);
    }

    #[test]
    fn every_legal_move_round_trips_through_its_action() {
        let fens = [
            None,
            Some("r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 9"),
            Some("r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R4RK1 b kq - 5 9"),
            Some("rnbqkbnr/ppp2ppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"),
            Some("8/1P2k3/8/8/8/8/4K1p1/8 w - - 0 1"),
            Some("8/1P2k3/8/8/8/8/4K1p1/8 b - - 0 1"),
            Some("7k/8/5K1R/8/8/8/8/8 b - - 0 1"),
        ];
        for fen in fens {
            let g = game(fen, "");
            let legal = LegalActions::of(&g);
            assert_eq!(legal.len(), g.legal_moves().len());
            let mask = legal.mask();
            assert_eq!(mask.iter().filter(|&&b| b).count(), legal.len(), "{fen:?}");
            let mut seen = std::collections::HashSet::new();
            for &(mv, action) in &legal.moves {
                assert!(seen.insert(action), "duplicate action {action} for {fen:?}");
                assert_eq!(
                    decode_action(g.position(), action),
                    Some(mv),
                    "{fen:?} {mv:?}"
                );
            }
        }
    }

    #[test]
    fn decode_rejects_off_board_and_illegal() {
        let g = game(None, "");
        assert_eq!(decode_action(g.position(), ACTION_SIZE), None);
        // a1 moving north-west leaves the board.
        assert_eq!(decode_action(g.position(), 7 * 7), None);
        // e2e4 is legal, e2e5 (distance 3) is not.
        assert_eq!(decode_action(g.position(), 12 * 73 + 2), None);
    }
}
