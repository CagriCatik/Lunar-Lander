#!/usr/bin/env bash
# Bootstrap without global pip installs, sudo, or shell-profile modifications.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
EXTRAS=""
case "${1:-}" in
  --dev) EXTRAS='[dev]' ;;
  '') ;;
  *) echo "Usage: $0 [--dev]" >&2; exit 2 ;;
esac
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN. Install Python 3.11–3.13, then retry." >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "Python 3.11–3.13 is required"'
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
# SWIG must exist before pip builds Box2D on platforms without a wheel.
"$VENV_DIR/bin/python" -m pip install 'swig==4.3.1'
export PATH="$VENV_DIR/bin:$PATH"
if ! "$VENV_DIR/bin/python" -m pip install -e ".$EXTRAS"; then
  echo "Install failed. For Box2D source builds, install a C++ toolchain:" >&2
  echo "  macOS: xcode-select --install" >&2
  echo "  Ubuntu/Debian: sudo apt install build-essential python3-dev python3-venv" >&2
  exit 1
fi
"$ROOT/scripts/doctor.sh"
echo "Ready. Run: ./scripts/run.sh --render"
