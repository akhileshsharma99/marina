#!/usr/bin/env bash
# A/B two configurations of the engine with fastchess and an SPRT.
#
# The baseline and the candidate are the same binary (or two binaries) with different UCI
# options; both play the same openings with colours swapped, node- or time-limited, and
# fastchess stops as soon as the sequential test decides. Elo and the SPRT are reported
# from the candidate's ("test") point of view: "H1 was accepted" means the candidate is
# at least elo1 better. Typical use:
#
#   tools/ab.sh --base "Batch=128" --test "Batch=256 BatchRamp=4" --nodes 800
#   tools/ab.sh --base "CPuct=150" --test "CPuct=125" --tc 10+0.1
#
# Requires `fastchess` on PATH (https://github.com/Disservin/fastchess) and an opening book
# (BOOK, PGN or EPD). The PGN written to OUT carries nodes, seldepth and nps per move.
#
# Environment / flags:
#   WEIGHTS         network for both sides: a name from nets.toml or a directory with
#                   model.json + model.safetensors (default: the embedded network)
#   BOOK            opening book, PGN or EPD (default: books/8moves-v3-seed0-256.pgn, 256
#                   positions sampled from official-stockfish/books 8moves_v3)
#   BINARY          engine binary (default: target/release/marina)
#   TEST_BINARY     a different binary for the candidate side (default: BINARY)
#   --binary PATH   the same, as flags (override the environment for one test)
#   --test-binary PATH
#   --base "K=V .." UCI options for the baseline
#   --test "K=V .." UCI options for the candidate
#   --nodes N       fixed nodes per move (mutually exclusive with --tc)
#   --tc B+I        time control in seconds, e.g. 10+0.1
#   --elo0/--elo1   SPRT hypotheses in Elo (default 0 / 5)
#   --rounds N      maximum rounds, two games each (default 5000)
#   --concurrency N games at a time (default 1; more shares the GPU and the result stops meaning anything)
#   --out FILE      PGN output (default: ab-<timestamp>.pgn)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BINARY="${BINARY:-$HERE/../target/release/marina}"
TEST_BINARY="${TEST_BINARY:-$BINARY}"
BOOK="${BOOK:-$HERE/books/8moves-v3-seed0-256.pgn}"
BASE_OPTS="" TEST_OPTS="" NODES="" TC="" ELO0=0 ELO1=5 ROUNDS=5000 CONCURRENCY=1 OUT=""
BINARY_FLAG="" TEST_BINARY_FLAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE_OPTS="$2"; shift 2 ;;
    --test) TEST_OPTS="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --tc) TC="$2"; shift 2 ;;
    --elo0) ELO0="$2"; shift 2 ;;
    --elo1) ELO1="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --binary) BINARY_FLAG="$2"; shift 2 ;;
    --test-binary) TEST_BINARY_FLAG="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
# A --binary flag stands in for both sides unless --test-binary names the candidate's.
[ -n "$BINARY_FLAG" ] && { BINARY="$BINARY_FLAG"; TEST_BINARY="$BINARY_FLAG"; }
[ -n "$TEST_BINARY_FLAG" ] && TEST_BINARY="$TEST_BINARY_FLAG"
if [ -n "$NODES" ] && [ -n "$TC" ]; then echo "--nodes and --tc are exclusive" >&2; exit 2; fi
if [ -z "$NODES" ] && [ -z "$TC" ]; then echo "one of --nodes or --tc is required" >&2; exit 2; fi
command -v fastchess >/dev/null || { echo "fastchess not on PATH" >&2; exit 2; }
[ -x "$BINARY" ] || { echo "engine binary $BINARY not found" >&2; exit 2; }
[ -x "$TEST_BINARY" ] || { echo "engine binary $TEST_BINARY not found" >&2; exit 2; }
[ -f "$BOOK" ] || { echo "opening book $BOOK not found" >&2; exit 2; }
OUT="${OUT:-ab-$(date +%Y%m%d-%H%M%S).pgn}"

# "K=V K=V" -> "option.K=V option.K=V"; the network from WEIGHTS unless the options name one:
# a nets.toml name is Network=<name>, a directory is Network=file WeightsFile=<dir>.
network_options() {
  case "$1" in
    "") ;;
    */*|.*) printf ' option.Network=file option.WeightsFile=%s' "$1" ;;
    *) printf ' option.Network=%s' "$1" ;;
  esac
}
options() {
  local out=""
  for kv in $1; do out="$out option.$kv"; done
  case "$1" in *Network=*|*WeightsFile=*) ;; *) out="$out$(network_options "${WEIGHTS:-}")" ;; esac
  printf '%s' "$out"
}
limit() { if [ -n "$NODES" ]; then printf 'nodes=%s' "$NODES"; else printf 'tc=%s' "$TC"; fi; }
case "$BOOK" in *.epd) FORMAT=epd ;; *) FORMAT=pgn ;; esac

echo "base: $BINARY $BASE_OPTS"
echo "test: $TEST_BINARY $TEST_OPTS"
echo "limit: $(limit)   sprt: elo0=$ELO0 elo1=$ELO1   out: $OUT"
# shellcheck disable=SC2046
# The candidate is listed first: fastchess reports Elo and runs the SPRT from the first
# engine's point of view, so a passing candidate reads "H1 was accepted", Elo positive.
exec fastchess \
  -engine cmd="$TEST_BINARY" name=test $(options "$TEST_OPTS") \
  -engine cmd="$BINARY" name=base $(options "$BASE_OPTS") \
  -each $(limit) \
  -openings file="$BOOK" format="$FORMAT" order=random \
  -repeat -rounds "$ROUNDS" -concurrency "$CONCURRENCY" \
  -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 model=normalized \
  -ratinginterval 20 \
  -pgnout file="$OUT" notation=uci nodes=true seldepth=true nps=true
