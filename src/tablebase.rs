//! Syzygy tablebases: exact results for positions with few pieces.
//!
//! Two probes. At a **leaf** the WDL value becomes a proof (win, draw or loss from the
//! side to move) that the tree propagates like a mate; outcomes the fifty-move rule takes
//! away (cursed wins, blessed losses) count as draws.
//! At the **root** the DTZ tables pick the move directly, ranked under the fifty-move rule
//! the way Stockfish's root probe does, so a won endgame is converted rather than
//! shuffled; the search is skipped.
//!
//! Positions with castling rights are never in the tables and are not probed.

use std::path::Path;

use shakmaty::{Chess, Move, Position};
use shakmaty_syzygy::AmbiguousWdl;
use thiserror::Error;

use crate::search::tree::Proof;

/// Ply distance recorded on tablebase proofs: far beyond any mate the search can find, so
/// a proven mate is preferred to a tablebase win; reported as a fixed tablebase score
/// rather than a mate distance.
pub const TB_PLIES: u16 = 10_000;

#[derive(Debug, Error)]
pub enum TablebaseError {
    #[error("cannot read Syzygy directory {path}: {source}")]
    Directory {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("no Syzygy tables found under {0}")]
    Empty(String),
}

pub struct Tablebase {
    inner: shakmaty_syzygy::Tablebase<Chess>,
    /// Probe positions with at most this many pieces (`SyzygyProbeLimit`, capped by the
    /// largest table present).
    probe_limit: usize,
    tables: usize,
}

impl Tablebase {
    /// Open every directory in `paths` (separated by the platform path separator).
    pub fn open(paths: &str, probe_limit: u32) -> Result<Self, TablebaseError> {
        let mut inner = shakmaty_syzygy::Tablebase::new();
        let mut tables = 0;
        for path in std::env::split_paths(paths) {
            if path.as_os_str().is_empty() {
                continue;
            }
            tables += inner.add_directory(Path::new(&path)).map_err(|source| {
                TablebaseError::Directory {
                    path: path.display().to_string(),
                    source,
                }
            })?;
        }
        if tables == 0 {
            return Err(TablebaseError::Empty(paths.to_string()));
        }
        let probe_limit = (probe_limit as usize).min(inner.max_pieces());
        Ok(Self {
            inner,
            probe_limit,
            tables,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "Syzygy: {} tables, up to {} pieces, probing at most {}",
            self.tables,
            self.inner.max_pieces(),
            self.probe_limit
        )
    }

    /// Whether the tables can say anything about `pos`.
    #[inline]
    pub fn covers(&self, pos: &Chess) -> bool {
        pos.board().occupied().count() <= self.probe_limit && pos.castles().is_empty()
    }

    /// Exact result of `pos` from its side to move, or `None` when the tables do not cover
    /// it. Wins and losses the fifty-move rule would take away are draws.
    pub fn probe(&self, pos: &Chess) -> Option<Proof> {
        if !self.covers(pos) {
            return None;
        }
        match self.inner.probe_wdl(pos) {
            Ok(AmbiguousWdl::Win) => Some(Proof::Win(TB_PLIES)),
            Ok(AmbiguousWdl::Loss) => Some(Proof::Loss(TB_PLIES)),
            Ok(_) => Some(Proof::Draw),
            // Castling rights, too many pieces, or a missing or unreadable table: search on
            // as if uncovered.
            Err(_) => None,
        }
    }

    /// The tablebase move at the root and the position's proof, when the tables cover it
    /// and agree on a move. The move is ranked by DTZ under the fifty-move rule (a
    /// zeroing move that keeps the win counts as progress; losses are delayed), so
    /// following it converts a won position.
    pub fn root(&self, pos: &Chess) -> Option<(Move, Proof)> {
        let proof = self.probe(pos)?;
        match self.inner.best_move(pos) {
            Ok(Some((mv, _dtz))) => Some((mv, proof)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Run with `MARINA_SYZYGY_PATH=<dir with 3-4-5 tables> cargo test tablebase`; skipped
    //! (passing) when the tables are not there.

    use super::*;
    use shakmaty::CastlingMode;
    use shakmaty::fen::Fen;

    fn tables() -> Option<Tablebase> {
        let path = std::env::var("MARINA_SYZYGY_PATH").ok()?;
        Some(Tablebase::open(&path, 5).expect("tables open"))
    }

    fn pos(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap()
    }

    #[test]
    fn leaf_probe_gives_exact_proofs() {
        let Some(tb) = tables() else { return };
        // KQ v K, white to move: win. Black to move in the same position: loss.
        assert_eq!(
            tb.probe(&pos("8/8/8/8/8/3k4/8/Q3K3 w - - 0 1")),
            Some(Proof::Win(TB_PLIES))
        );
        assert_eq!(
            tb.probe(&pos("8/8/8/8/8/3k4/8/Q3K3 b - - 0 1")),
            Some(Proof::Loss(TB_PLIES))
        );
        // KB v K is a draw whoever moves.
        assert_eq!(
            tb.probe(&pos("8/8/8/8/8/3k4/8/B3K3 w - - 0 1")),
            Some(Proof::Draw)
        );
        // Too many pieces for the limit, or castling rights: not probed.
        assert_eq!(tb.probe(&pos("k7/8/8/8/8/8/8/1RR1K2R w - - 0 1")), None);
        assert_eq!(tb.probe(&pos("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")), None);
    }

    #[test]
    fn root_probe_picks_a_converting_move() {
        let Some(tb) = tables() else { return };
        // KP v K with the pawn one step from promotion: the tablebase move must promote.
        let position = pos("8/3P4/8/8/8/8/8/k6K w - - 0 1");
        let (mv, proof) = tb.root(&position).expect("covered");
        assert_eq!(proof, Proof::Win(TB_PLIES));
        assert!(mv.is_promotion(), "{mv:?}");
    }

    #[test]
    fn missing_directory_is_an_error() {
        assert!(Tablebase::open("/nonexistent/syzygy", 5).is_err());
    }
}
