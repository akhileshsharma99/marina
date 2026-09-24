#!/usr/bin/env sh
# Installs the latest Marina release binary to /usr/local/bin (override with INSTALL_DIR):
#
#   curl -fsSL https://raw.githubusercontent.com/akhileshsharma99/marina/main/install.sh | sh
#
# Picks the asset for this machine: Linux x64 (the CUDA build when an NVIDIA driver is
# present; it falls back to the CPU on machines without one), Linux arm64, macOS arm64.
# Windows: download marina-windows-x64.exe from the releases page. VERSION=vX.Y.Z installs
# that release instead of the latest; MARINA_CPU=1 forces the CPU build on Linux.
#
# The download is checked against the .sha256 published beside it. Both come from the same
# release, so this catches a corrupt or truncated download, not a compromised release; for
# that, verify the binary's provenance with `gh attestation verify <file> --repo <repo>`.
set -eu

main() {
  REPO="akhileshsharma99/marina"
  INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
  BIN="marina"

  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os" in
    Linux)
      case "$arch" in
        x86_64|amd64)
          asset="marina-linux-x64"
          if [ -z "${MARINA_CPU:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
            asset="marina-linux-x64-cuda"
          fi ;;
        aarch64|arm64) asset="marina-linux-arm64" ;;
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
      esac ;;
    Darwin)
      case "$arch" in
        arm64) asset="marina-macos-arm64" ;;
        *) echo "macOS builds are for Apple silicon; build from source on Intel Macs (cargo build --release)" >&2; exit 1 ;;
      esac ;;
    *) echo "unsupported OS: $os (Windows: download marina-windows-x64.exe from https://github.com/$REPO/releases)" >&2; exit 1 ;;
  esac

  if [ -n "${VERSION:-}" ]; then
    url="https://github.com/$REPO/releases/download/$VERSION/$asset"
  else
    url="https://github.com/$REPO/releases/latest/download/$asset"
  fi

  tmp="$(mktemp)"
  trap 'rm -f "$tmp" "$tmp.sha256"' EXIT
  if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fSL --proto '=https' --tlsv1.2 --progress-bar "$1" -o "$2"; }
  elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q --https-only --show-progress -O "$2" "$1"; }
  else
    echo "neither curl nor wget is installed; download $url yourself" >&2
    exit 1
  fi
  echo "downloading $asset..."
  if ! fetch "$url" "$tmp"; then
    echo "download failed; see https://github.com/$REPO/releases for the available binaries" >&2
    exit 1
  fi
  if ! fetch "$url.sha256" "$tmp.sha256"; then
    echo "checksum download failed ($url.sha256)" >&2
    exit 1
  fi
  expected="$(awk '{print $1}' "$tmp.sha256")"
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$tmp" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "$tmp" | awk '{print $1}')"
  fi
  if [ "$expected" != "$actual" ]; then
    echo "sha256 mismatch for $asset: expected $expected, got $actual" >&2
    exit 1
  fi

  if [ -d "$INSTALL_DIR" ] && [ -w "$INSTALL_DIR" ]; then
    install -m 0755 "$tmp" "$INSTALL_DIR/$BIN"
  elif mkdir -p "$INSTALL_DIR" 2>/dev/null && [ -w "$INSTALL_DIR" ]; then
    install -m 0755 "$tmp" "$INSTALL_DIR/$BIN"
  else
    echo "installing to $INSTALL_DIR needs sudo"
    sudo mkdir -p "$INSTALL_DIR"
    sudo install -m 0755 "$tmp" "$INSTALL_DIR/$BIN"
  fi
  echo "installed $INSTALL_DIR/$BIN"
  "$INSTALL_DIR/$BIN" --version
}

main "$@"
