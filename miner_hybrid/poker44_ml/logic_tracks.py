"""Handcrafted bot-family logic tracks for Poker44 chunks.

These trackers encode poker/bot intuition as [0, 1] signals. They are consumed as
ML features (primary path) and can optionally provide a rule prior for blending.

Bot families targeted (subtype not labeled in data — behavioral only):
  * script / repeat-line bots
  * fixed sizing-grid bots
  * passive caller bots
  * hyper-aggressive bots
  * shallow / short-tree bots
  * low-entropy / autopilot preflop bots
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _mean(values: list[float]) -> float:
    return _safe_div(sum(values), len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return math.sqrt(max(0.0, _mean([(v - m) * (v - m) for v in values])))


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return _safe_div(ent, math.log(len(counts)))


def _action_types(hand: dict[str, Any]) -> list[str]:
    return [
        str((a or {}).get("action_type") or "").lower().strip()
        for a in (hand.get("actions") or [])
        if isinstance(a, dict)
    ]


def _pot_fractions(hand: dict[str, Any]) -> list[float]:
    fracs: list[float] = []
    for action in hand.get("actions") or []:
        if not isinstance(action, dict):
            continue
        amount = _safe_float(action.get("amount"), 0.0)
        pot_before = _safe_float(action.get("pot_before"), 0.0)
        if amount <= 0.0 or pot_before <= 0.0:
            continue
        fracs.append(amount / pot_before)
    return fracs


def _near_grid(value: float, targets: tuple[float, ...] = (0.33, 0.50, 0.66, 0.75, 1.0), tol: float = 0.06) -> bool:
    return any(abs(value - t) <= tol for t in targets)


def hand_logic_tracks(hand: dict[str, Any]) -> dict[str, float]:
    """Per-hand logic signals in [0, 1]."""
    actions = hand.get("actions") or []
    streets = hand.get("streets") or []
    types = _action_types(hand)
    meaningful = [t for t in types if t in {"call", "check", "bet", "raise", "fold"}]
    counts = Counter(meaningful)
    n = max(1, len(meaningful))
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    passive = counts.get("call", 0) + counts.get("check", 0)
    fold = counts.get("fold", 0)
    fracs = _pot_fractions(hand)
    grid_hits = sum(1 for v in fracs if _near_grid(v))
    street_depth = _safe_div(len(streets), 3.0)
    action_len = _safe_div(len(actions), 16.0)

    return {
        "logic_hand_passive": _clamp01(_safe_div(passive, n)),
        "logic_hand_aggression": _clamp01(_safe_div(aggressive, n)),
        "logic_hand_fold": _clamp01(_safe_div(fold, n)),
        "logic_hand_low_entropy": _clamp01(1.0 - _entropy(meaningful)),
        "logic_hand_sizing_grid": _clamp01(_safe_div(grid_hits, max(1, len(fracs)))),
        "logic_hand_shallow_tree": _clamp01(1.0 - street_depth),
        "logic_hand_short_actions": _clamp01(1.0 - min(1.0, action_len)),
        "logic_hand_preflop_only": float(
            bool(streets) and all(str(s).lower() == "preflop" for s in streets)
        ),
    }


def chunk_logic_tracks(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Chunk-level bot-family tracks + a fused rule prior."""
    if not chunk:
        return {
            "logic_repeat_line_rate": 0.0,
            "logic_signature_unique_inv": 0.0,
            "logic_size_grid_score": 0.0,
            "logic_entropy_collapse": 0.0,
            "logic_passive_caller": 0.0,
            "logic_hyper_aggro": 0.0,
            "logic_shallow_tree": 0.0,
            "logic_preflop_autopilot": 0.0,
            "logic_consistency": 0.0,
            "logic_rule_prior": 0.5,
            "logic_rule_confidence": 0.0,
        }

    per_hand = [hand_logic_tracks(hand) for hand in chunk]
    n = float(len(chunk))

    action_sigs: list[tuple[str, ...]] = []
    preflop_sigs: list[tuple[str, ...]] = []
    all_types: list[str] = []
    for hand in chunk:
        types = tuple(_action_types(hand))
        action_sigs.append(types)
        preflop = tuple(
            str((a or {}).get("action_type") or "").lower().strip()
            for a in (hand.get("actions") or [])
            if isinstance(a, dict) and str((a or {}).get("street") or "").lower() == "preflop"
        )
        preflop_sigs.append(preflop)
        all_types.extend(types)

    top_share = _safe_div(max(Counter(action_sigs).values()), n)
    unique_share = _safe_div(len(set(action_sigs)), n)
    preflop_top = _safe_div(max(Counter(preflop_sigs).values()), n) if preflop_sigs else 0.0

    passive_mean = _mean([h["logic_hand_passive"] for h in per_hand])
    aggro_mean = _mean([h["logic_hand_aggression"] for h in per_hand])
    grid_mean = _mean([h["logic_hand_sizing_grid"] for h in per_hand])
    shallow_mean = _mean([h["logic_hand_shallow_tree"] for h in per_hand])
    short_mean = _mean([h["logic_hand_short_actions"] for h in per_hand])
    low_ent_mean = _mean([h["logic_hand_low_entropy"] for h in per_hand])
    fold_mean = _mean([h["logic_hand_fold"] for h in per_hand])

    aggro_std = _std([h["logic_hand_aggression"] for h in per_hand])
    passive_std = _std([h["logic_hand_passive"] for h in per_hand])
    consistency = _clamp01(1.0 - 0.5 * (aggro_std + passive_std) / 0.35)

    entropy_collapse = _clamp01(0.55 * low_ent_mean + 0.45 * (1.0 - _entropy(all_types)))
    passive_caller = _clamp01(0.60 * passive_mean + 0.25 * (1.0 - fold_mean) + 0.15 * (1.0 - aggro_mean))
    hyper_aggro = _clamp01(0.70 * aggro_mean + 0.30 * (1.0 - passive_mean))
    shallow_tree = _clamp01(0.55 * shallow_mean + 0.45 * short_mean)
    size_grid = _clamp01(grid_mean)
    repeat_line = _clamp01(top_share)
    preflop_autopilot = _clamp01(0.6 * preflop_top + 0.4 * _mean([h["logic_hand_preflop_only"] for h in per_hand]))

    # Fused rule prior: emphasize script/sizing/entropy tells that kings rely on.
    rule_prior = _clamp01(
        0.22 * repeat_line
        + 0.18 * size_grid
        + 0.16 * entropy_collapse
        + 0.14 * passive_caller
        + 0.12 * hyper_aggro
        + 0.10 * shallow_tree
        + 0.08 * preflop_autopilot
    )
    # Confidence rises when tracks agree (high consistency + extreme prior).
    extremity = abs(rule_prior - 0.5) * 2.0
    rule_confidence = _clamp01(0.55 * consistency + 0.45 * extremity)

    return {
        "logic_repeat_line_rate": repeat_line,
        "logic_signature_unique_inv": _clamp01(1.0 - unique_share),
        "logic_size_grid_score": size_grid,
        "logic_entropy_collapse": entropy_collapse,
        "logic_passive_caller": passive_caller,
        "logic_hyper_aggro": hyper_aggro,
        "logic_shallow_tree": shallow_tree,
        "logic_preflop_autopilot": preflop_autopilot,
        "logic_consistency": consistency,
        "logic_rule_prior": rule_prior,
        "logic_rule_confidence": rule_confidence,
        # Keep useful hand aggregates as explicit ML features too.
        "logic_passive_mean": _clamp01(passive_mean),
        "logic_aggression_mean": _clamp01(aggro_mean),
        "logic_fold_mean": _clamp01(fold_mean),
        "logic_low_entropy_mean": _clamp01(low_ent_mean),
    }


def blend_ml_with_logic(
    ml_score: float,
    logic_prior: float,
    logic_confidence: float,
    *,
    max_blend: float = 0.35,
) -> float:
    """Soft gate: clear logic cases pull ML score toward the rule prior."""
    conf = _clamp01(logic_confidence)
    # Only blend when confidence is material; ambiguous chunks stay ML-dominated.
    weight = max_blend * conf * conf
    return _clamp01((1.0 - weight) * float(ml_score) + weight * float(logic_prior))
