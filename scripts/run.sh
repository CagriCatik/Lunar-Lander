#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" -m lunar_mpc.cli run --config "$ROOT/configs/default.toml" "$@"
