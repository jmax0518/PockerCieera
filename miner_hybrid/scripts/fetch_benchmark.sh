#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$REPO:${PYTHONPATH:-}"
DAYS="${DAYS:-7}"
python3 training/fetch_benchmark.py --out-dir "$ROOT/data/benchmark" --days "$DAYS" "$@"
