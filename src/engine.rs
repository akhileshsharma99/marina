//! The long-lived engine state and the search lifecycle.
//!
//! One [`Engine`] outlives every UCI command: options, the current game, and the search
//! thread. A search runs on its own thread with a clone of the game and a shared
//! [`StopSignal`]; it reports back through a channel of [`SearchEvent`]s, which the UCI
//! loop writes to the GUI. The loop thread therefore never blocks on the search (except to
//! join it at `quit`), and `isready`, `stop` and `quit` are answered while a search runs.

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::net::{self, HashNetwork, Network};
use crate::options::{Backend, Options};
use crate::position::Game;
use crate::search::time::TimeState;
use crate::search::{self, SavedTree, Shared, StopSignal};
use crate::tablebase::{Tablebase, TablebaseError};
use crate::uci::command::GoLimits;

/// What a running search sends to the UCI loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchEvent {
    /// A rendered `info ...` line (without the newline).
    Info(String),
    /// The network stopped answering; the `bestmove` that follows is the best move found
    /// before it did, and the engine should not search again.
    Failed(String),
    /// The search finished; exactly one per `go`.
    BestMove {
        best: String,
        ponder: Option<String>,
    },
}

pub struct Engine {
    pub options: Options,
    pub game: Game,
    /// The position of the last search that was not a ponder (the start of the game until
    /// then). A `position` that does not continue it begins a new game for the clock
    /// bookkeeping; the pondered position and the one reached after a ponder miss both
    /// continue it, so neither resets the bank.
    anchor: Game,
    /// The network searches evaluate with; [`Engine::ensure_network`] replaces the
    /// start-up placeholder before the first search.
    pub network: Arc<dyn Network>,
    /// `(WeightsFile, Backend, Device)` the current network was loaded with; `None` until
    /// the first load.
    loaded: Option<(String, Backend, net::Device)>,
    /// How many networks have been loaded: 0 is the placeholder, every successful load
    /// counts one more. Each search runs under the generation current at its start and
    /// tags the tree it leaves with it, so a search that was still running on the old
    /// network when the GUI switched cannot hand its tree to the first search on the new
    /// one (it stores the tree after `forget_tree` has run).
    generation: u64,
    /// Syzygy tables, when `SyzygyPath` names some.
    pub tablebase: Option<Arc<Tablebase>>,
    /// `(SyzygyPath, SyzygyProbeLimit)` last applied, whether the tables opened or not, so
    /// a failing path is reported once rather than on every `isready`.
    loaded_tablebase: Option<(String, u32)>,
    /// Clock bookkeeping across the moves of the current game.
    time_state: Arc<Mutex<TimeState>>,
    /// The last search's tree, for the next search to continue from.
    tree: Arc<Mutex<Option<SavedTree>>>,
    stop: Arc<StopSignal>,
    /// Set by `go`, cleared when its `bestmove` is observed.
    searching: bool,
    search: Option<JoinHandle<()>>,
    /// A `go` that arrived while the previous search was still delivering its `bestmove`
    /// (GUIs send `stop`, `position`, `go` back to back, and analysis GUIs do it on every
    /// move the user steps through); it starts when that `bestmove` has been sent. Only
    /// the latest is kept: it is the position the GUI is looking at.
    pending: Option<Pending>,
}

struct Pending {
    limits: GoLimits,
    /// A `stop` came in the meantime, meant for this search.
    stopped: bool,
}

/// What became of a `go`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Go {
    Started,
    /// Starts once the running search has sent its `bestmove`.
    Queued,
    /// Took the place of a `go` that was queued and had not started: that one never runs
    /// and gets no `bestmove`, which is what a GUI stepping through moves wants.
    Replaced,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let options = Options::default();
        Self {
            game: Game::new(shakmaty::CastlingMode::Standard),
            anchor: Game::new(shakmaty::CastlingMode::Standard),
            options,
            network: Arc::new(HashNetwork::default()),
            loaded: None,
            generation: 0,
            tablebase: None,
            loaded_tablebase: None,
            time_state: Arc::new(Mutex::new(TimeState::default())),
            tree: Arc::new(Mutex::new(None)),
            stop: Arc::new(StopSignal::new()),
            searching: false,
            search: None,
            pending: None,
        }
    }

    /// `ucinewgame`: forget the game, the clock state and the tree.
    pub fn new_game(&mut self) {
        self.game = Game::new(shakmaty::CastlingMode::Standard);
        self.anchor = self.game.clone();
        self.reset_time_state();
        self.forget_tree();
    }

    /// `position`: a new game starts here unless it continues the game being played (GUIs
    /// are not required to send `ucinewgame`), in which case the clock bookkeeping is
    /// reset too.
    pub fn set_position(&mut self, game: Game) {
        if game.continuation_from(&self.anchor).is_none() {
            self.reset_time_state();
            self.anchor = game.clone();
        }
        self.game = game;
    }

    fn reset_time_state(&self) {
        *self
            .time_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = TimeState::default();
    }

    fn forget_tree(&self) {
        *self
            .tree
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    /// Make the network match `WeightsFile`, `Backend`, `Threads` and `Batch`, loading it
    /// if any changed since the last load. Called on `isready` and before `go`, the points where the GUI
    /// expects the engine to have applied its options. Returns the description of a newly
    /// loaded network. A network that fails to load is an error, never a substitute: the
    /// UCI loop reports it and quits.
    pub fn ensure_network(&mut self) -> Result<Option<String>, net::LoadError> {
        let device = net::Device {
            threads: self.options.threads as usize,
            max_batch: self.options.batch() as usize,
            precision: self.options.precision,
        };
        let wanted = (self.options.weights_spec(), self.options.backend, device);
        if self.loaded.as_ref() == Some(&wanted) {
            return Ok(None);
        }
        let network = net::resolve(&wanted.0, wanted.1, device)?;
        let description = format!("{}: {}", self.options.weights_label(), network.describe());
        self.network = Arc::from(network);
        self.loaded = Some(wanted);
        self.generation += 1;
        // The tree's values came from the old network. Forgetting it here frees the memory
        // at once; the generation is what keeps a tree stored later by a search still
        // running on the old network from being reused.
        self.forget_tree();
        Ok(Some(description))
    }

    /// Make the tablebases match `SyzygyPath` and `SyzygyProbeLimit`, (re)opening them if
    /// either changed. Called alongside [`Engine::ensure_network`]. Returns a description
    /// when tables were opened or dropped.
    pub fn ensure_tablebase(&mut self) -> Result<Option<String>, TablebaseError> {
        let wanted = (
            self.options.syzygy_path.clone(),
            self.options.syzygy_probe_limit,
        );
        if self.loaded_tablebase.as_ref() == Some(&wanted) {
            return Ok(None);
        }
        if wanted.0.trim().is_empty() {
            let had = self.tablebase.take().is_some();
            self.loaded_tablebase = Some(wanted);
            return Ok(had.then(|| "Syzygy: tables closed".to_string()));
        }
        let opened = Tablebase::open(&wanted.0, wanted.1);
        self.loaded_tablebase = Some(wanted);
        let tablebase = opened?;
        let description = tablebase.describe();
        self.tablebase = Some(Arc::new(tablebase));
        Ok(Some(description))
    }

    /// Whether a search is running: from `go` until its `bestmove` has been observed
    /// ([`Engine::search_finished`]), not until its thread has unwound, so a `go` that
    /// follows the `bestmove` immediately is never refused.
    pub fn is_searching(&self) -> bool {
        self.searching
    }

    /// The `bestmove` of the running search has been received: join its thread. A queued
    /// `go` is now due ([`Engine::has_pending`], [`Engine::start_pending`]).
    pub fn search_finished(&mut self) {
        self.searching = false;
        self.reap();
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Start the queued `go`, stopped at once if a `stop` meant for it has already come.
    pub fn start_pending(&mut self, events: Sender<SearchEvent>) {
        if let Some(pending) = self.pending.take() {
            self.start(pending.limits, events);
            if pending.stopped {
                self.stop.stop();
            }
        }
    }

    /// Start a search on its own thread, or queue it behind the one still running. Every
    /// search that starts sends exactly one `bestmove`; a queued `go` that is superseded
    /// before it starts sends none. A `stop` received for the superseded one does not
    /// carry over: the GUI asked for the new search.
    pub fn go(&mut self, limits: GoLimits, events: Sender<SearchEvent>) -> Go {
        if self.searching {
            let replaced = self.pending.is_some();
            self.pending = Some(Pending {
                limits,
                stopped: false,
            });
            return if replaced { Go::Replaced } else { Go::Queued };
        }
        self.start(limits, events);
        Go::Started
    }

    fn start(&mut self, limits: GoLimits, events: Sender<SearchEvent>) {
        self.reap();
        self.stop.reset();
        self.searching = true;
        if !limits.ponder {
            self.anchor = self.game.clone();
        }
        let game = self.game.clone();
        let stop = Arc::clone(&self.stop);
        let options = self.options.clone();
        let shared = Shared {
            network: Arc::clone(&self.network),
            generation: self.generation,
            tablebase: self.tablebase.clone(),
            time_state: Arc::clone(&self.time_state),
            tree: Arc::clone(&self.tree),
        };
        self.search = Some(std::thread::spawn(move || {
            search::run(game, limits, options, shared, stop, events);
        }));
    }

    /// Ask the running search to finish now; it still sends its `bestmove`. With a `go`
    /// queued, the GUI means that one too: it will stop as soon as it starts.
    pub fn stop(&mut self) {
        self.stop.stop();
        if let Some(pending) = self.pending.as_mut() {
            pending.stopped = true;
        }
    }

    /// `ponderhit`: the pondering search goes on as a normal search of the same position,
    /// its clock running from now (a queued `go ponder` starts as a plain `go`). Nothing
    /// happens when no search is pondering.
    pub fn ponderhit(&mut self) {
        match self.pending.as_mut() {
            Some(pending) => pending.limits.ponder = false,
            None if self.searching => self.stop.ponderhit(),
            None => {}
        }
    }

    /// Stop and wait for the search thread to exit (used on `quit`).
    pub fn shutdown(&mut self) {
        self.stop();
        self.reap();
    }

    fn reap(&mut self) {
        if let Some(handle) = self.search.take() {
            let _ = handle.join();
        }
    }
}
