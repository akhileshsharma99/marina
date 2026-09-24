# Changelog

## 0.1.0 (2026-09-24)

Initial release.

### Features

* **marina:** batched PUCT search over a transformer that gives a policy over the legal moves and a win/draw/loss estimate; two embedded networks, `small` (38M parameters, the default) and `nano` (2.7M), selected with the `Network` option, or any exported network directory through `WeightsFile`
* **marina:** CUDA backend (fp16, fp8 on Ada and newer), Metal backend for Apple silicon (fp16 or fp32), and a CPU backend; `Backend` and `Precision` choose, `auto` picks the fastest available
* **marina:** Syzygy tablebase probing (`SyzygyPath`, `SyzygyProbeLimit`), `MultiPV`, `UCI_ShowWDL`, pondering, and `ScoreType` for the centipawn scale (`centipawn` or `lc0`)
* **marina:** search settings tuned per network (`CPuct`, `FPUReduction`, `Batch`), overridable as options
* **marina:** binaries for Linux (x64, x64 CUDA, arm64), macOS (Apple silicon) and Windows, each with a checksum and build provenance; an install script; a container image (`ghcr.io/akhileshsharma99/marina`); every release ships the networks it embeds and the games behind the README's strength numbers
