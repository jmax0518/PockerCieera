# Poker44 Scoring Logic — Bug Report & Reward-Optimization Notes

Status: informal engineering analysis, based on reading and executing
`poker44/score/scoring.py` and `poker44/validator/forward.py` on this checkout.

Not an official Poker44 disclosure. Findings below are reproducible against the
code as committed in this repo — see [Reproduction](#reproduction) for a script
that prints the exact numbers cited here.

## Summary

The validator reward function (`poker44/score/scoring.py::reward`) is intended
to be a rank-first metric with a hard "does this miner actually flag bots
without hurting humans" safety gate. In practice it has several defects:

| # | Defect | Severity | Type |
|---|---|---|---|
| 1 | Single-class evaluation windows award a reward floor (`0.35`–`0.70`) regardless of prediction quality | High | Logic bug |
| 2 | `latency_quality` is a hardcoded constant; `LATENCY_WEIGHT` (5%) is never actually earned, it's given away | Medium | Dead code / mislabeled weight |
| 3 | `calibration_quality` is an alias of `human_safety_penalty`, not an independent metric | Medium | Mislabeled weight |
| 4 | `threshold_sanity_quality` is flat at `1.0` for any FPR in `[0%, 10%]` — no incentive to be safer than "just under 10%" | Low/Medium | Design smell |
| 5 | Any `risk_scores` length mismatch discards the entire response for that cycle (no partial credit); prediction/label buffers grow unbounded (never trimmed) | Low | Design smell / resource leak |
| 6 | Winner ties are broken in favor of the lowest UID | Low | Fairness smell |

Bug 1 is the most significant: it is a genuine logic defect in `reward()`
itself, not merely a data/operational issue, even though production
operational practice (`require_mixed`) reduces how often it is actually
triggered in practice.

---

## Bug 1 — Single-class windows award a reward floor unrelated to prediction quality

### Root cause

```76:81:poker44/score/scoring.py
if positive_count <= 0 or negative_count <= 0:
    threshold_sanity_quality = 1.0
elif true_positives <= 0:
    threshold_sanity_quality = 0.0
```

`_threshold_metrics` checks "is one class entirely absent from this window?"
**before** checking whether the miner caught anything. When a class is
missing:

- `threshold_sanity_quality` is forced to `1.0` (best possible), regardless of
  `hard_fpr` or `true_positives`.
- `_recall_at_fpr` (used for the `bot_recall` term) explicitly returns
  `(0.0, 0.0)` when `positive_count <= 0` or `negative_count <= 0`.
- `average_precision_score` (used for `ap_score`) is mathematically `1.0`
  whenever there are zero negative examples, because precision is always
  `TP / (TP + FP) = TP / TP = 1` when `FP` can never be nonzero. This holds
  **regardless of the miner's actual scores.**

Since `calibration_quality = human_safety_penalty = threshold_sanity_quality`
(see [Bug 3](#bug-3--calibration_quality-is-an-alias-not-an-independent-metric)),
a single-class window collapses `base_score` to one of two possible floors:

- **All-bot window** (`negative_count == 0`): `ap_score = 1.0`,
  `bot_recall = 0.0`, `threshold_sanity_quality = 1.0` →
  `base_score = 0.35·1 + 0.30·0 + 0.20·1 + 0.10·1 + 0.05·1 = 0.70`
- **All-human window** (`positive_count == 0`): `ap_score = 0.0`,
  `bot_recall = 0.0`, `threshold_sanity_quality = 1.0` →
  `base_score = 0.35·0 + 0.30·0 + 0.20·1 + 0.10·1 + 0.05·1 = 0.35`

**Both floors are awarded no matter what the miner predicts.**

### Empirical proof

Run directly against `poker44/score/scoring.py::reward` (see
[Reproduction](#reproduction) for the full script):

| Scenario | Miner behavior | `reward()` |
|---|---|---|
| All-bot window (5 bots, 0 humans) | Scores everything `0.0` — catches **zero** bots | **0.70** |
| All-human window (5 humans, 0 bots) | Scores everything `0.51` — flags **every** human as a bot | **0.35** |
| All-human window (5 humans, 0 bots) | Scores everything `1.0` — max risk on every human | **0.35** |
| Mixed window (3 humans, 2 bots), control | Scores everything `1.0` — same "flag everyone" behavior | **0.00** |

```text
--- Case 1: all-bot window, miner predicts ZERO for everything ---
reward = 0.7000000000000001

--- Case 2: all-human window, miner predicts 0.51 for EVERYTHING ---
reward = 0.35000000000000003
{'hard_fpr': 1.0, 'threshold_sanity_quality': 1.0, ...}

--- Case 4 (control): MIXED window, miner predicts 1.0 for EVERYTHING ---
reward = 0.0
{'hard_fpr': 1.0, 'threshold_sanity_quality': 0.0, ...}
```

Case 2 and Case 4 use the **identical** "flag everyone" strategy. The only
variable is whether a human happened to be mixed into the labeled window —
that alone swings the reward from `0.35` to `0.0`. This proves the defect is
in the scoring function's handling of degenerate input, not in any specific
miner's behavior.

### Why this isn't fully closed by operational practice

```122:154:poker44/validator/runtime_provider.py
require_mixed: bool = True
...
require_mixed=_env_bool("POKER44_PROVIDER_REQUIRE_MIXED", True),
```

The validator sends a `requireMixed: true` hint when asking the backend to
publish an active chunk via `/internal/eval/publish-current`. This reduces how
often Bug 1 triggers in production, but:

1. It is a **backend-side best-effort** request — the backend implementation
   is not in this repo and enforcement can't be verified here.
2. It is only sent when `attempt_publish_current` runs with a valid
   `POKER44_PROVIDER_INTERNAL_SECRET`. The validator docs explicitly say
   operators may leave that secret unset and rely on backend auto-publish
   instead — in which case this hint is **never sent**.
3. It is not checked anywhere inside `scoring.py`. The reward function itself
   has no defense in depth against a homogeneous window; it fully trusts its
   caller.

### Suggested fix

`reward()` should treat a single-class window as **no signal**, not as a
free pass:

```python
if positive_count <= 0 or negative_count <= 0:
    # No basis to evaluate bot detection or human safety this window.
    return 0.0, {..., "degenerate_window": True}
```

or, if a non-zero reward is intentional for degenerate windows (e.g. to avoid
starving miners during data gaps), the floor should be small and explicitly
documented, not silently inherited from taking the "best case" branch of a
per-metric fallback.

---

## Bug 2 — `latency_quality` is a hardcoded constant

```108:109:poker44/score/scoring.py
calibration_quality = human_safety_penalty
latency_quality = 1.0
```

`LATENCY_WEIGHT = 0.05` is applied in `base_score`, but `latency_quality`
never reads the miner's actual response time (`dendrite.process_time`, which
*is* captured elsewhere in `forward.py` as `latency_mean_seconds` and stored in
the metrics dict — it just never reaches `reward()`). Every miner receives the
full 5% regardless of speed, as long as it responds inside the query timeout
(enforced separately, not by this term).

**Impact:** 5% of total possible reward is unconditional. There is currently
no way for a slow-but-otherwise-correct miner to lose reward due to latency,
nor for a fast miner to gain extra reward for it.

---

## Bug 3 — `calibration_quality` is an alias, not an independent metric

```107:108:poker44/score/scoring.py
human_safety_penalty = threshold_metrics["threshold_sanity_quality"]
calibration_quality = human_safety_penalty
```

`calibration_quality` is not computed from anything calibration-specific
(e.g. Brier score, expected calibration error, reliability curve) — it is
literally the same value as `human_safety_penalty`. As a result,
`HUMAN_SAFETY_WEIGHT (20%) + CALIBRATION_WEIGHT (10%) = 30%` of total reward
collapses into **one single step-function metric**
(`threshold_sanity_quality`), despite being presented as two independently
weighted components.

**Impact:** there is no reward signal for genuine probabilistic calibration
(e.g., a model whose `0.7` output really does correspond to ~70% bot
likelihood) beyond clearing the coarse `hard_fpr <= 0.10` / `true_positives >
0` gate.

---

## Bug 4 — The safety gate is a flat plateau up to 10% FPR

```78:83:poker44/score/scoring.py
elif hard_fpr <= 0.10:
    threshold_sanity_quality = 1.0
else:
    threshold_sanity_quality = max(0.0, 1.0 - (hard_fpr - 0.10) / 0.90)
```

Empirically:

```text
fpr=9%:  threshold_sanity_quality = 1.0     reward = 0.5342
fpr=11%: threshold_sanity_quality = 0.9889  reward = 0.5133
```

There is **zero** reward difference between a human false-positive rate of
`0%` and `9.99%`. The decay past `10%` is also gentle (`11%` FPR only drops
quality to `0.99`). Combined with Bug 3, this flat plateau controls 30% of
total reward.

**Secondary inconsistency:** the `bot_recall` term uses a stricter `max_fpr =
0.05` cutoff (`_recall_at_fpr(scores, labels, max_fpr=0.05)`), while the hard
gate tolerates FPR up to `0.10`. Two different tolerances exist for
ostensibly the same "don't hurt humans" concern, at two different weights.

**Impact:** a rational miner is incentivized to push its decision threshold
right up against the 10% FPR line to harvest extra recall, since there is no
cost to doing so below that line — the metric doesn't reward being safer than
required.

---

## Bug 5 — All-or-nothing coverage; unbounded buffer growth

```203:214:poker44/validator/forward.py
if len(scores_f) != len(chunks):
    bt.logging.warning(...)
    response_metadata[uid] = {"coverage_rate": 0.0, ...}
    validator.coverage_buffer.setdefault(uid, []).append(0.0)
    continue
```

Any length mismatch between `risk_scores` and the number of chunks sent
discards the **entire** response — there is no partial credit for, say, 95 of
100 correctly-shaped scores.

Separately, `prediction_buffer`, `label_buffer`, `coverage_buffer`, and
`latency_buffer` (initialized in `neurons/validator.py`) are only ever
appended to via `.extend()` / `.append()`; nothing in the codebase trims them.
They are read with `buf[-window:]` in `_compute_windowed_rewards`, so scoring
correctness isn't affected, but the dictionaries grow for the entire lifetime
of the validator process — an unbounded memory-growth issue on long-running
validators.

---

## Bug 6 — Winner ties favor the lowest UID

```705:706:poker44/validator/forward.py
sorted_rewards = sorted(reward_map.items(), key=lambda item: (-item[1], item[0]))
winner_uid, winner_reward = sorted_rewards[0]
```

Because scoring is winner-take-all (`WINNER_TAKE_ALL = True`,
`BURN_FRACTION = 0.0` in `poker44/validator/constants.py`), exact ties matter.
Bug 1 means multiple miners can land on the exact same floor value (`0.35` or
`0.70`) in a degenerate window purely by chance. Ties are resolved in favor of
the numerically lowest UID, a small structural bias toward earlier-registered
miners.

---

## Practical implications: how the current formula can be gamed / optimized

These follow directly from the defects above and don't require breaking any
rule — they're the rational response to the reward function as written.

1. **The 0.5 hard-gate is binary, not continuous, up to 10% FPR** (Bug 4).
   Since it controls 30% of reward (Bug 3) and gives identical credit at 0%
   and 9.99% FPR, there is no benefit to conservative calibration below that
   line — a miner should push its decision boundary to use the full FPR
   budget in exchange for recall.
2. **`bot_recall` is checked at a stricter 5% FPR bar than the gate's 10%** —
   optimize rank-separation broadly (AP + recall@5%FPR = 65% of reward)
   rather than only tuning the threshold placement.
3. **Latency is free** (Bug 2) — no need to optimize inference speed beyond
   staying within the query timeout.
4. **Never return a mis-sized `risk_scores` list** — it zeroes that cycle's
   coverage *and* skips buffer accumulation entirely, which can also stall a
   miner below the window size `_compute_windowed_rewards` requires to score
   it at all (`reward = 0.0` if `len(pred_buf) < window`). Reliability is an
   implicit prerequisite even though there's no explicit uptime term.
5. **Single-class windows (Bug 1) are not something a miner can trigger**, but
   they do mean observed reward volatility across cycles doesn't purely
   reflect prediction quality — a lazy and a skilled miner can look identical
   on those specific cycles.

---

## Reproduction

Run from the repo root (`Poker44-subnet/`) with the project's virtualenv
active:

```bash
python3 -c "
import numpy as np
from poker44.score.scoring import reward

print('--- Case 1: all-bot window, miner predicts ZERO for everything ---')
labels = np.array([1,1,1,1,1])
scores = np.array([0.0,0.0,0.0,0.0,0.0])
val, metrics = reward(scores, labels)
print('reward =', val)
print(metrics)

print()
print('--- Case 2: all-human window, miner predicts 0.51 for EVERYTHING ---')
labels = np.array([0,0,0,0,0])
scores = np.array([0.51,0.51,0.51,0.51,0.51])
val, metrics = reward(scores, labels)
print('reward =', val)
print(metrics)

print()
print('--- Case 4 (control): MIXED window, miner predicts 1.0 for EVERYTHING ---')
labels = np.array([0,0,0,1,1])
scores = np.array([1.0,1.0,1.0,1.0,1.0])
val, metrics = reward(scores, labels)
print('reward =', val)
print(metrics)

print()
print('--- Case 5: mixed window, hard_fpr near the 10% boundary ---')
labels = np.array([0]*100 + [1]*10)
scores = np.array([0.6]*9 + [0.1]*91 + [0.6]*10)
val, metrics = reward(scores, labels)
print('fpr=9%:', metrics['hard_fpr'], 'quality=', metrics['threshold_sanity_quality'], 'reward=', val)

scores = np.array([0.6]*11 + [0.1]*89 + [0.6]*10)
val, metrics = reward(scores, labels)
print('fpr=11%:', metrics['hard_fpr'], 'quality=', metrics['threshold_sanity_quality'], 'reward=', val)
"
```

---

## Files referenced

- `poker44/score/scoring.py` — reward function under analysis
- `poker44/validator/forward.py` — windowed reward aggregation, buffer
  management, winner selection
- `poker44/validator/constants.py` — `WINNER_TAKE_ALL`, `BURN_FRACTION`
- `poker44/validator/runtime_provider.py` — `require_mixed` mitigation
- `neurons/validator.py` — buffer initialization
- `tests/test_scoring.py` — existing test coverage (does not cover
  single-class windows; all fixtures use mixed-label arrays)
