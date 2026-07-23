#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$ROOT:$REPO:${PYTHONPATH:-}"
export POKER44_MODEL_PATH="${POKER44_MODEL_PATH:-$ROOT/models/hybrid_v1.joblib}"

WALLET_NAME="${WALLET_NAME:?set WALLET_NAME}"
HOTKEY="${HOTKEY:?set HOTKEY}"
AXON_PORT="${AXON_PORT:-8091}"
NETUID="${NETUID:-126}"
NETWORK="${NETWORK:-finney}"

ARGS=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
)

if [[ -n "${ALLOWED_VALIDATOR_HOTKEYS:-}" ]]; then
  # shellcheck disable=SC2206
  KEYS=($ALLOWED_VALIDATOR_HOTKEYS)
  ARGS+=(--blacklist.allowed_validator_hotkeys "${KEYS[@]}")
else
  ARGS+=(--blacklist.force_validator_permit)
fi

exec python3 "$ROOT/neurons/miner.py" "${ARGS[@]}"
