<div align="center">

# Marina

**A transformer chess engine that gets top-engine strength out of a small network.**

[![CI](https://github.com/akhileshsharma99/marina/actions/workflows/ci.yml/badge.svg)](https://github.com/akhileshsharma99/marina/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/akhileshsharma99/marina?logo=github)](https://github.com/akhileshsharma99/marina/releases/latest)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

</div>

Marina is a UCI engine: batched PUCT search over a transformer that gives a policy over the legal moves and a win/draw/loss estimate for every position. The default network (`small`) has 38M parameters, the other (`nano`) 2.7M, and the search is built to get the most out of every evaluation. Runs on NVIDIA GPUs (CUDA), Apple GPUs (Metal) or the CPU; the binaries embed both networks.

## Strength

Matches against Stockfish 17.1 at full strength under CCRL's conditions, 200 games per match. Scores are Marina's, W-D-L, and link to the games, which ship with every release.

**Blitz, 2m+1s**

| network | result | Elo vs Stockfish 17.1 | CCRL Blitz estimate |
| --- | --- | --- | --- |
| `nano` (2.7M) | [0-72-128 (18.0%)](https://github.com/akhileshsharma99/marina/releases/latest/download/nano-vs-stockfish-17.1-blitz.pgn) | -263 ± 39 | 3507 ± 40 |
| `small` (38M) | [1-142-57 (36.0%)](https://github.com/akhileshsharma99/marina/releases/latest/download/small-vs-stockfish-17.1-blitz.pgn) | -100 ± 24 | 3670 ± 26 |

**Hyperbullet, 10s+0.1s**

| network | result | Elo vs Stockfish 17.1 |
| --- | --- | --- |
| `nano` (2.7M) | [1-49-150 (12.8%)](https://github.com/akhileshsharma99/marina/releases/latest/download/nano-vs-stockfish-17.1-hyperbullet.pgn) | -334 ± 48 |
| `small` (38M) | [0-104-96 (26.0%)](https://github.com/akhileshsharma99/marina/releases/latest/download/small-vs-stockfish-17.1-hyperbullet.pgn) | -182 ± 31 |

Conditions: Stockfish on one thread with Hash 256, 6-man Syzygy tablebases for both sides, one game at a time, colours swapped on every opening. Stockfish runs on one core of a Threadripper 7960X, Marina on an RTX 4090.
The CCRL estimate is Stockfish 17.1's CCRL Blitz rating (3770 ± 9, single CPU, list of September 21, 2026) plus the measured difference, the two uncertainties combined, and it is an underestimate: CCRL's clock is equivalent to 2m+1s on an Intel i7-4770K, and Stockfish gets the full clock on a much faster core here, so it plays stronger than its listed rating. An official rating needs a CCRL submission.
Hyperbullet has no CCRL list - it is used for faster testing and iteration.

## Getting started

```sh
curl -fsSL https://raw.githubusercontent.com/akhileshsharma99/marina/main/install.sh | sh
```

Linux (picks the CUDA build when an NVIDIA driver is present; it runs on the CPU without one) and macOS on Apple silicon, where the engine runs on the GPU through Metal.  
Or download a binary from the [releases page](https://github.com/akhileshsharma99/marina/releases/latest): [`marina-linux-x64`](https://github.com/akhileshsharma99/marina/releases/latest/download/marina-linux-x64), [`marina-linux-x64-cuda`](https://github.com/akhileshsharma99/marina/releases/latest/download/marina-linux-x64-cuda), [`marina-linux-arm64`](https://github.com/akhileshsharma99/marina/releases/latest/download/marina-linux-arm64), [`marina-macos-arm64`](https://github.com/akhileshsharma99/marina/releases/latest/download/marina-macos-arm64), [`marina-windows-x64.exe`](https://github.com/akhileshsharma99/marina/releases/latest/download/marina-windows-x64.exe).

Then point any UCI GUI (En Croissant, Cute Chess, Arena, ...) at it, or:

```sh
marina
> uci
> position startpos
> go movetime 1000
```

Or run the container image. It uses the GPU when started with `--gpus all` and the CPU otherwise; the engine talks UCI on stdin/stdout, so `-i` is what a GUI or match runner needs.

```sh
docker run --rm -i --gpus all ghcr.io/akhileshsharma99/marina
docker run --rm -i --gpus all -v /path/to/syzygy:/syzygy ghcr.io/akhileshsharma99/marina   # then: setoption name SyzygyPath value /syzygy
```

Build from source with Rust 1.88+. The first build fetches the networks (~165 MB) from the [`nets` release](https://github.com/akhileshsharma99/marina/releases/tag/nets) and checks their hashes against `nets.toml`.

```sh
cargo build --release                     # CPU backend
cargo build --release --features cuda     # + CUDA (needs the CUDA toolkit)
```

`cargo test`, `marina bench`, `marina verify`, `tools/ab.sh` and the CUDA build image (`docker build --target dev .`) are described in [CONTRIBUTING.md](CONTRIBUTING.md).

## Options

The options every UCI GUI knows:

| option | default | what it does |
| --- | --- | --- |
| `DebugLogFile` | | transcript of every protocol line, in and out |
| `MoveOverhead` | 20 | per-move slack for the GUI round trip, ms |
| `MultiPV` | 1 | root moves to report an `info` line for |
| `Ponder` | false | lets the GUI send `go ponder` |
| `SyzygyPath` | | tablebase directories, separated by the platform path separator |
| `SyzygyProbeLimit` | 5 | probe positions with at most this many pieces |
| `Threads` | 1 | threads for the CPU backend (the GPU backends ignore it) |
| `UCI_ShowWDL` | false | `wdl <win> <draw> <loss>` in permille on every `info` line |

Marina's own:

| option | default | what it does |
| --- | --- | --- |
| `Backend` | `auto` | `auto` picks CUDA when this build has it and a device answers, Metal on macOS, else `cpu` |
| `Batch` | per network (128) | leaves evaluated per network call |
| `BatchRamp` | 4 | a batch asks for at most `tree size / BatchRamp` leaves; 0 turns the ramp off |
| `CollisionBudget` | 100 | stop filling a batch once the visits it could not place exceed this percentage of it; 0 = no limit |
| `CPuct` | per network (`small` 175, `nano` 150) | PUCT exploration constant, in hundredths |
| `FPUReduction` | per network (25) | first-play urgency, in hundredths |
| `InFlight` | 2 | batches queued at the network at once |
| `Network` | `small` | which embedded network plays: `small`, `nano`, or `file` for the directory in `WeightsFile` |
| `Precision` | `auto` | arithmetic of the linear layers: `auto` is the fastest the backend has (fp8 on CUDA with an Ada or newer card, fp16 on Metal, fp32 on the CPU); `fp32`, `fp16` or `fp8` to insist, falling back to the nearest the backend supports |
| `ScoreType` | `centipawn` | how `score cp` is derived from the value head: `centipawn` (the logistic scale, +100 is about 64% expected score) or `lc0` (Leela's tangent mapping); `UCI_ShowWDL` prints the probabilities either way |
| `WeightsFile` | | a directory with `model.json` + `model.safetensors`, used when `Network` is `file` |

The search options (`Batch` through `InFlight`) are tuned by SPRT; their defaults are what the engine plays with. Chess960 is not supported.

## Acknowledgements

- [Lichess](https://database.lichess.org/#evals) for the evaluation database the networks are trained on, and [Syzygy](https://syzygy-tables.info) tablebases for exact endgame labels
- [shakmaty](https://github.com/niklasf/shakmaty) and [shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy) for move generation and tablebase probing
- [Reckless](https://github.com/codedeliveryservice/Reckless) and [Stockfish](https://github.com/official-stockfish/Stockfish), the opponents every change is measured against
- [fastchess](https://github.com/Disservin/fastchess) for running the matches, and the opening book from [official-stockfish/books](https://github.com/official-stockfish/books)
- [Leela Chess Zero](https://lczero.org) and [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) for the search this engine builds on

## License

Copyright (C) 2026 Akhilesh Sharma. This project is licensed under the [GNU General Public License v3.0 or later](LICENSE). The networks (the `nets` release and the archive in every version release) are distributed under the same terms.
