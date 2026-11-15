#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" -m pytest "$@"
"$PYTHON" -m ruff check src tests
