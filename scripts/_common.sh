#!/usr/bin/env bash
# Sourced by entry points, so they also work outside the repository directory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Environment missing. Run $ROOT/scripts/setup.sh first." >&2
  exit 1
fi
export PYGAME_HIDE_SUPPORT_PROMPT=1
