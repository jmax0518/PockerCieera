#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HYB="$ROOT/miner_hybrid"
cd "$ROOT"

# Bittensor >=10.5 defaults BT_NO_PARSE_CLI_ARGS=true (ignores CLI). Force parse ON.
export BT_NO_PARSE_CLI_ARGS=0
export PYTHONPATH="$HYB:$ROOT:${PYTHONPATH:-}"
export POKER44_MODEL_PATH="${POKER44_MODEL_PATH:-$HYB/models/hybrid_v1.joblib}"

# Public model identity (Poker44 manifest / dashboard)
export POKER44_MODEL_OPEN_SOURCE=1
export POKER44_MODEL_REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/jmax0518/PockerCieera}"
export POKER44_MODEL_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-4eb7d21e3239de40a6f7a9c460fa32078e6081ce}"
export POKER44_MODEL_NAME="${POKER44_MODEL_NAME:-poker44-hybrid-v1}"
export POKER44_MODEL_VERSION="${POKER44_MODEL_VERSION:-1.2.0-safety}"

# Also set env fallbacks used by poker44.utils.config._ensure_neuron_config
export WALLET_NAME="${WALLET_NAME:-VPS-3-SECURE}"
export HOTKEY="${HOTKEY:-cierra-poker}"
export NETUID="${NETUID:-126}"
export NETWORK="${NETWORK:-finney}"

AXON_PORT="${AXON_PORT:-8126}"
EXTERNAL_IP="${EXTERNAL_IP:-176.9.86.107}"

PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

exec "$PY" "$HYB/neurons/miner.py" \
  --netuid "$NETUID" \
  --subtensor.network "$NETWORK" \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$HOTKEY" \
  --axon.ip 0.0.0.0 \
  --axon.port "$AXON_PORT" \
  --axon.external_ip "$EXTERNAL_IP" \
  --axon.external_port "$AXON_PORT" \
  --logging.info \
  --blacklist.force_validator_permit
