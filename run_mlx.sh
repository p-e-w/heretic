#!/usr/bin/env bash
# Run heretic with the MLX backend on an MLX model.
# Usage: ./run_mlx.sh /path/to/mlx-model [extra heretic args...]

set -euo pipefail

MODEL="${1:?Usage: $0 /path/to/mlx-model [--n-trials 10 ...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${HERETIC_PYTHON:-/opt/homebrew/bin/python3.11}"

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$PYTHON" -c "
import sys
sys.argv = ['heretic', '--backend', 'mlx', '--model', '$MODEL'] + sys.argv[1:]
from heretic.main import main
main()
" "$@"
