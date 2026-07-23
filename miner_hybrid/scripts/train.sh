#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$REPO:${PYTHONPATH:-}"

# CPU-only training. No CUDA/GPU required for hybrid-v1 tree stack.
echo "Training hybrid-v1 on CPU (GPU not required)..."
python3 training/train_hybrid.py \
  --data-dir "$ROOT/data/benchmark" \
  --out "$ROOT/models/hybrid_v1.joblib" \
  "$@"
