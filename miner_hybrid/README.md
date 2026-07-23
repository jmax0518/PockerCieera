# Poker44 hybrid miner (CPU tree stack)

Hybrid of cold-king n-grams/stack/remap + hot-king sklearn reliability.
Optional mild safety budget (off by default).

## Do you need a GPU?

**No.** Hybrid v1 trains and serves on CPU:
- ExtraTrees, RandomForest, HistGradientBoosting (+ optional LightGBM)
- Logistic meta-learner + isotonic calibration + threshold_logit remap

GPU would only matter later if you enable a deep sequence model (Set Transformer), which this v1 intentionally skips.

Typical laptop/server CPU training on ~1 week of benchmark data: minutes to low tens of minutes depending on cores.

## Architecture

1. Features: schema stats + signatures + fixed action n-grams (from UID 142/85)
2. **Logic tracks**: handcrafted bot-family priors (`logic_tracks.py`) wired into features
3. Robust feature filter for live OOD (from cold)
4. OOF stacked trees → logistic meta → blended isotonic (from cold)
5. `threshold_logit_v1` score remap tuned to official `reward()` (from cold)
6. Soft logic/ML blend when rule confidence is high (clear cases)
7. Optional mild `topk` safety budget (from hot, default OFF)

Logic tracks include: repeat-line, sizing-grid, entropy-collapse, passive-caller,
hyper-aggro, shallow-tree, preflop-autopilot, consistency, fused `logic_rule_prior`.

Disable soft blend at serve time:

```bash
export POKER44_LOGIC_BLEND=0
```


## Setup

From repo root:

```bash
source .venv/bin/activate   # or miner_env
pip install -r miner_hybrid/requirements.txt
```

## Train (CPU)

```bash
cd miner_hybrid
chmod +x scripts/*.sh
./scripts/fetch_benchmark.sh          # downloads public labeled benchmark
./scripts/train.sh                    # writes models/hybrid_v1.joblib
```

Useful flags:

```bash
./scripts/train.sh --holdout-dates 2 --human-weight 1.3
./scripts/train.sh --enable-safety-budget --safety-fraction 0.25
```

## Run miner

```bash
WALLET_NAME=my_cold \
HOTKEY=my_hot \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="hk1 hk2" \
./scripts/run_miner.sh
```

Model path override:

```bash
export POKER44_MODEL_PATH=/abs/path/to/hybrid_v1.joblib
```

## Layout

```text
miner_hybrid/
  poker44_ml/     features, stack, inference, calibration
  training/       fetch + train
  neurons/miner.py
  models/         hybrid_v1.joblib (after train)
  scripts/
```

## Notes

- Live eval labels stay private; train only on the public benchmark API.
- Keep your served code and published manifest commit honest — high scores get audited.
- Start without safety budget; enable only if live FPR spikes.
