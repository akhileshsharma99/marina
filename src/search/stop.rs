//! The signals shared by the UCI loop and a running search: `stop` and `ponderhit`.
//!
//! The search polls them: the gather loop checks [`StopSignal::is_set`] once per batch (an
//! atomic load, no syscall) and the UCI loop sets it on `stop` or `quit`. A pondering
//! search also polls [`StopSignal::is_ponderhit`], which turns it into a normal search of
//! the same position. Nothing blocks on either, so they are flags and nothing more.

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Default)]
pub struct StopSignal {
    stop: AtomicBool,
    ponderhit: AtomicBool,
}

impl StopSignal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request a stop.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// The pondered move was played: a pondering search continues under its limits.
    pub fn ponderhit(&self) {
        self.ponderhit.store(true, Ordering::SeqCst);
    }

    /// Arm for a new search.
    pub fn reset(&self) {
        self.stop.store(false, Ordering::SeqCst);
        self.ponderhit.store(false, Ordering::SeqCst);
    }

    /// Cheap check for hot loops.
    #[inline]
    pub fn is_set(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn is_ponderhit(&self) -> bool {
        self.ponderhit.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_ponderhit_and_reset() {
        let signal = StopSignal::new();
        assert!(!signal.is_set() && !signal.is_ponderhit());
        signal.stop();
        assert!(signal.is_set());
        signal.ponderhit();
        assert!(signal.is_ponderhit());
        signal.reset();
        assert!(!signal.is_set() && !signal.is_ponderhit());
    }
}
