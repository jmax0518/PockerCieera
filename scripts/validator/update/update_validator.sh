#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

PROCESS_NAME="${PROCESS_NAME:-poker44_validator}"
WALLET_NAME="${WALLET_NAME:-}"
WALLET_HOTKEY="${WALLET_HOTKEY:-}"
NETUID="${NETUID:-126}"
SUBTENSOR_PARAM="${SUBTENSOR_PARAM:---subtensor.network finney}"
VALIDATOR_ENV_DIR="${VALIDATOR_ENV_DIR:-validator_env}"
VALIDATOR_EXTRA_ARGS="${VALIDATOR_EXTRA_ARGS:-}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
VALIDATOR_SCRIPT="${VALIDATOR_SCRIPT:-./neurons/validator.py}"
SAFE_BITTENSOR_CLI_VERSION="${SAFE_BITTENSOR_CLI_VERSION:-9.23.2}"
SAFE_BITTENSOR_WALLET_VERSION="${SAFE_BITTENSOR_WALLET_VERSION:-4.1.0}"
SAFE_BITTENSOR_SDK_MIN_VERSION="${SAFE_BITTENSOR_SDK_MIN_VERSION:-10.3.0}"
SAFE_BITTENSOR_SDK_MAX_MAJOR="${SAFE_BITTENSOR_SDK_MAX_MAJOR:-11}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ "$VALIDATOR_SCRIPT" != /* ]]; then
  VALIDATOR_SCRIPT="$REPO_ROOT/${VALIDATOR_SCRIPT#./}"
fi

if [ -x "$REPO_ROOT/$VALIDATOR_ENV_DIR/bin/python" ]; then
  PYTHON_BIN="$REPO_ROOT/$VALIDATOR_ENV_DIR/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Error: no Python interpreter found" >&2
  exit 1
fi

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Branch: $TARGET_BRANCH"
echo "[INFO] Process: $PROCESS_NAME"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Current commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

install_runtime_bittensor_stack() {
  echo "[INFO] Installing runtime-431 compatible Bittensor SDK, CLI, and wallet versions..."
  "$PYTHON_BIN" -m pip uninstall -y scalecodec >/dev/null 2>&1 || true
  "$PYTHON_BIN" -m pip install \
    "bittensor>=${SAFE_BITTENSOR_SDK_MIN_VERSION},<${SAFE_BITTENSOR_SDK_MAX_MAJOR}" \
    "bittensor-cli==${SAFE_BITTENSOR_CLI_VERSION}" \
    "bittensor-wallet==${SAFE_BITTENSOR_WALLET_VERSION}"
  "$PYTHON_BIN" -m pip uninstall -y scalecodec >/dev/null 2>&1 || true
  "$PYTHON_BIN" -m pip install --force-reinstall "cyscale==0.5.0"
}

guard_runtime_bittensor_stack() {
  echo "[INFO] Verifying runtime-431 Bittensor package versions..."
  "$PYTHON_BIN" - <<'PY'
from importlib import metadata
from os import environ
from sys import exit

safe_cli = environ.get("SAFE_BITTENSOR_CLI_VERSION", "9.23.2")
safe_wallet = environ.get("SAFE_BITTENSOR_WALLET_VERSION", "4.1.0")
safe_sdk_min_raw = environ.get("SAFE_BITTENSOR_SDK_MIN_VERSION", "10.3.0")
safe_sdk_max_major = int(environ.get("SAFE_BITTENSOR_SDK_MAX_MAJOR", "11"))
blocked = {
    "bittensor-cli": "9.18.2",
    "bittensor-wallet": "4.0.2",
}

packages = {}
for name in ("bittensor", "bittensor-cli", "bittensor-wallet", "scalecodec"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

def version_tuple(value):
    parts = []
    for raw in str(value or "").replace("-", ".").split("."):
        if raw.isdigit():
            parts.append(int(raw))
        else:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

safe_sdk_min = version_tuple(safe_sdk_min_raw)
for name, blocked_version in blocked.items():
    if packages[name] == blocked_version:
        print(f"Blocked version installed: {name}=={blocked_version}")
        exit(1)

sdk_version = version_tuple(packages["bittensor"])
if packages["bittensor"] is None or sdk_version < safe_sdk_min or sdk_version[0] >= safe_sdk_max_major:
    print(
        "Bittensor SDK version is not compatible with the Bittensor 431 runtime:",
        f"bittensor=={packages['bittensor']}",
        f"required: bittensor>={safe_sdk_min_raw},<{safe_sdk_max_major}",
    )
    exit(1)

if packages["scalecodec"] is not None:
    print(
        "Legacy scalecodec is installed and conflicts with the runtime-431 compatible cyscale stack.",
        "Run: pip uninstall -y scalecodec && pip install --force-reinstall cyscale==0.5.0",
    )
    exit(1)

if packages["bittensor-cli"] != safe_cli or packages["bittensor-wallet"] != safe_wallet:
    print(
        "Unexpected Bittensor package versions:",
        f"bittensor=={packages['bittensor']}",
        f"bittensor-cli=={packages['bittensor-cli']}",
        f"bittensor-wallet=={packages['bittensor-wallet']}",
    )
    exit(1)

print(
    "Verified pinned Bittensor package versions:",
    f"bittensor=={packages['bittensor']}",
    f"bittensor-cli=={packages['bittensor-cli']}",
    f"bittensor-wallet=={packages['bittensor-wallet']}",
)
PY
}

pushd "$REPO_ROOT" > /dev/null
git config --local core.fileMode false || true

AUTO_UPDATE_STASH_CREATED=0
AUTO_UPDATE_STASH_REF=""
if [ -n "$(git status --porcelain)" ]; then
  echo "[WARN] Local changes detected; stashing before update."
  git stash push --include-untracked -m "poker44-auto-update-prepull" >/dev/null
  AUTO_UPDATE_STASH_CREATED=1
  AUTO_UPDATE_STASH_REF="$(git stash list | head -n1 | cut -d: -f1)"
fi

echo "[INFO] Fetching latest Poker44 code from origin/$TARGET_BRANCH..."
git fetch origin "$TARGET_BRANCH"
git merge --ff-only "origin/$TARGET_BRANCH"
echo "[INFO] Updated commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [ "$AUTO_UPDATE_STASH_CREATED" = "1" ] && [ -n "$AUTO_UPDATE_STASH_REF" ]; then
  echo "[INFO] Restoring stashed local changes..."
  if ! git stash pop "$AUTO_UPDATE_STASH_REF"; then
    echo "[WARN] Could not automatically reapply stashed local changes; leaving stash for manual review."
  fi
fi
popd > /dev/null

if [ -x "$REPO_ROOT/$VALIDATOR_ENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/$VALIDATOR_ENV_DIR/bin/activate"
fi

echo "[INFO] Installing/updating Python dependencies..."
if [ -f "$REPO_ROOT/requirements.txt" ]; then
  "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements.txt"
fi
"$PYTHON_BIN" -m pip install -e "$REPO_ROOT"
install_runtime_bittensor_stack
guard_runtime_bittensor_stack

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

echo "[INFO] Restarting PM2 process '$PROCESS_NAME'..."
if ! pm2 restart "$PROCESS_NAME" --update-env; then
  if [ -z "$WALLET_NAME" ] || [ -z "$WALLET_HOTKEY" ]; then
    echo "[ERROR] PM2 process '$PROCESS_NAME' not found and wallet params are missing." >&2
    echo "[ERROR] Set WALLET_NAME and WALLET_HOTKEY to create the validator process." >&2
    exit 1
  fi

  read -r -a SUBTENSOR_ARG_ARRAY <<< "$SUBTENSOR_PARAM"
  VALIDATOR_CMD=(
    "$VALIDATOR_SCRIPT"
    --netuid "$NETUID"
    --wallet.name "$WALLET_NAME"
    --wallet.hotkey "$WALLET_HOTKEY"
    --logging.debug
  )
  VALIDATOR_CMD+=("${SUBTENSOR_ARG_ARRAY[@]}")

  if [ -n "$VALIDATOR_EXTRA_ARGS" ]; then
    read -r -a EXTRA_ARG_ARRAY <<< "$VALIDATOR_EXTRA_ARGS"
    VALIDATOR_CMD+=("${EXTRA_ARG_ARRAY[@]}")
  fi

  echo "[WARN] PM2 restart failed; starting a new Poker44 validator process"
  pm2 start "$PYTHON_BIN" --name "$PROCESS_NAME" -- "${VALIDATOR_CMD[@]}"
fi

echo "[INFO] Poker44 validator update completed"
