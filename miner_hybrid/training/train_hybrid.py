"""Train hybrid Poker44 detector (CPU tree stack + n-grams + score remap).

No GPU required. Uses HistGradientBoosting + ExtraTrees + RandomForest,
optionally LightGBM if installed. Meta-learner is logistic regression on OOF
base scores. Calibration mirrors current subnet reward().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402
from poker44_ml.calibration import BlendedIsotonicCalibrator  # noqa: E402
from poker44_ml.features import chunk_features  # noqa: E402
from poker44_ml.stacked import StackedEnsemble  # noqa: E402
from training.robust_features import filter_robust_feature_names  # noqa: E402


def _sanitize_chunk_group(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [prepare_hand_for_miner(hand) for hand in group]


def load_benchmark_rows(data_dir: Path) -> tuple[list[list[dict]], list[int], list[str]]:
    chunks_out: list[list[dict]] = []
    labels: list[int] = []
    dates: list[str] = []
    files = sorted(data_dir.glob("benchmark_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark_*.json under {data_dir}. Run fetch_benchmark.py first.")

    for path in files:
        payload = json.loads(path.read_text())
        source_date = str(payload.get("sourceDate") or path.stem.replace("benchmark_", ""))
        for item in payload.get("chunks") or []:
            groups = item.get("chunks") or []
            truths = item.get("groundTruth") or []
            if len(groups) != len(truths):
                continue
            for group, label in zip(groups, truths):
                if not group:
                    continue
                chunks_out.append(_sanitize_chunk_group(list(group)))
                labels.append(int(label))
                dates.append(source_date)
    return chunks_out, labels, dates


def build_feature_matrix(
    chunks: list[list[dict]],
    *,
    robust_only: bool = True,
) -> tuple[np.ndarray, list[str]]:
    feature_dicts = []
    for chunk in chunks:
        feats = chunk_features(chunk)
        feats["hand_count"] = float(len(chunk))
        feature_dicts.append(feats)
    all_names = sorted({name for row in feature_dicts for name in row})
    names = filter_robust_feature_names(all_names) if robust_only else all_names
    if len(names) < 32:
        raise RuntimeError(f"Too few features after filter: {len(names)}")
    matrix = np.asarray(
        [[float(row.get(name, 0.0)) for name in names] for row in feature_dicts],
        dtype=np.float64,
    )
    return matrix, names


def _make_base_models(random_state: int = 42) -> list[Any]:
    models: list[Any] = [
        ExtraTreesClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state + 1,
        ),
        HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=350,
            random_state=random_state + 2,
        ),
    ]
    try:
        from lightgbm import LGBMClassifier

        models.append(
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=48,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=random_state + 3,
                n_jobs=-1,
                verbosity=-1,
            )
        )
        print("LightGBM enabled")
    except Exception:
        print("LightGBM not installed; continuing with sklearn ensemble only")
    return models


def _clone_unfitted(model: Any) -> Any:
    params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    return model.__class__(**params)


def oof_stack(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[StackedEnsemble, np.ndarray]:
    base_templates = _make_base_models(random_state=random_state)
    n_models = len(base_templates)
    oof = np.zeros((len(y), n_models), dtype=float)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(x, y), start=1):
        print(f"  fold {fold_idx}/{n_folds} train={len(train_idx)} valid={len(valid_idx)}")
        for model_idx, template in enumerate(base_templates):
            model = _clone_unfitted(template)
            try:
                model.fit(
                    x[train_idx],
                    y[train_idx],
                    sample_weight=sample_weight[train_idx],
                )
            except TypeError:
                model.fit(x[train_idx], y[train_idx])
            proba = model.predict_proba(x[valid_idx])[:, 1]
            oof[valid_idx, model_idx] = proba

    meta = LogisticRegression(max_iter=1000, class_weight=None)
    meta.fit(oof, y, sample_weight=sample_weight)
    stacked_oof = meta.predict_proba(oof)[:, 1]

    # Refit bases on full data for deployment.
    fitted_bases: list[Any] = []
    for template in base_templates:
        model = _clone_unfitted(template)
        try:
            model.fit(x, y, sample_weight=sample_weight)
        except TypeError:
            model.fit(x, y)
        fitted_bases.append(model)

    calibrator = BlendedIsotonicCalibrator(blend=0.5)
    calibrator.fit(stacked_oof, y)
    ensemble = StackedEnsemble(
        base_models=fitted_bases,
        meta_model=meta,
        calibrator=calibrator,
        score_shift=0.0,
    )
    calibrated_oof = np.asarray(calibrator.transform(stacked_oof), dtype=float)
    return ensemble, calibrated_oof


def tune_score_remap(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    temperatures: list[float] | None = None,
) -> dict[str, Any]:
    temperatures = temperatures or [0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.75]
    human = scores[labels == 0]
    bot = scores[labels == 1]
    if human.size == 0 or bot.size == 0:
        return {"kind": "threshold_logit_v1", "threshold": 0.5, "temperature": 0.25}

    candidate_thresholds = sorted(
        {
            float(np.quantile(human, q))
            for q in (0.70, 0.80, 0.90, 0.95, 1.0)
        }
        | {
            float(np.quantile(bot, q))
            for q in (0.05, 0.10, 0.20, 0.30)
        }
        | {0.5}
    )

    best: dict[str, Any] | None = None
    best_reward = -1.0
    for threshold in candidate_thresholds:
        for temperature in temperatures:
            remapped = 1.0 / (
                1.0 + np.exp(-(scores - threshold) / max(temperature, 1e-6))
            )
            rew, metrics = reward(remapped, labels)
            if rew > best_reward:
                best_reward = float(rew)
                best = {
                    "kind": "threshold_logit_v1",
                    "threshold": float(threshold),
                    "temperature": float(temperature),
                    "tuned_reward": float(rew),
                    "tuned_ap": float(metrics.get("ap_score", 0.0)),
                    "tuned_bot_recall": float(metrics.get("bot_recall", 0.0)),
                    "tuned_hard_fpr": float(metrics.get("hard_fpr", 0.0)),
                }
    assert best is not None
    return best


def date_holdout_split(
    dates: list[str],
    *,
    holdout_dates: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(dates))
    if len(unique) <= holdout_dates:
        # Fall back to last 20% by index.
        n = len(dates)
        cut = max(1, int(n * 0.8))
        train_idx = np.arange(0, cut)
        test_idx = np.arange(cut, n)
        return train_idx, test_idx
    holdout = set(unique[-holdout_dates:])
    train_idx = np.asarray([i for i, d in enumerate(dates) if d not in holdout], dtype=int)
    test_idx = np.asarray([i for i, d in enumerate(dates) if d in holdout], dtype=int)
    return train_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid Poker44 miner model")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "benchmark"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "models" / "hybrid_v1.joblib"),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--human-weight", type=float, default=1.3)
    parser.add_argument("--holdout-dates", type=int, default=2)
    parser.add_argument("--no-robust-filter", action="store_true")
    parser.add_argument(
        "--enable-safety-budget",
        action="store_true",
        help="Stamp mild topk safety budget into artifact (default off)",
    )
    parser.add_argument("--safety-fraction", type=float, default=0.25)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Loading benchmark from {data_dir}")
    chunks, labels_list, dates = load_benchmark_rows(data_dir)
    y = np.asarray(labels_list, dtype=int)
    print(f"rows={len(y)} bots={int(y.sum())} humans={int((y == 0).sum())} dates={len(set(dates))}")

    print("Building features (schema + n-grams + logic tracks)...")
    x_all, feature_names = build_feature_matrix(
        chunks, robust_only=not args.no_robust_filter
    )
    logic_n = sum(1 for n in feature_names if n.startswith("logic_"))
    print(f"features={len(feature_names)} logic_tracks={logic_n} shape={x_all.shape}")

    train_idx, test_idx = date_holdout_split(dates, holdout_dates=args.holdout_dates)
    print(
        f"split train={len(train_idx)} test={len(test_idx)} "
        f"holdout_dates={sorted(set(dates[i] for i in test_idx))}"
    )

    sample_weight = np.ones(len(train_idx), dtype=float)
    sample_weight[y[train_idx] == 0] *= float(args.human_weight)

    print("Training OOF stack on CPU (no GPU needed)...")
    ensemble, oof_scores = oof_stack(
        x_all[train_idx],
        y[train_idx],
        sample_weight,
        n_folds=max(2, args.n_folds),
    )

    # Evaluate OOF before remap.
    oof_reward, oof_metrics = reward(oof_scores, y[train_idx])
    print(
        f"OOF calibrated reward={oof_reward:.4f} "
        f"ap={oof_metrics['ap_score']:.4f} "
        f"recall@5%fpr={oof_metrics['bot_recall']:.4f} "
        f"hard_fpr={oof_metrics['hard_fpr']:.4f}"
    )

    print("Tuning threshold_logit score_remap...")
    score_remap = tune_score_remap(oof_scores, y[train_idx])
    print(f"score_remap={score_remap}")

    def _remap(scores: np.ndarray) -> np.ndarray:
        thr = float(score_remap["threshold"])
        temp = max(float(score_remap["temperature"]), 1e-6)
        return 1.0 / (1.0 + np.exp(-(scores - thr) / temp))

    remapped_oof = _remap(oof_scores)
    remapped_reward, remapped_metrics = reward(remapped_oof, y[train_idx])
    print(
        f"OOF remapped reward={remapped_reward:.4f} "
        f"ap={remapped_metrics['ap_score']:.4f} "
        f"hard_bot_recall={remapped_metrics['hard_bot_recall']:.4f} "
        f"hard_fpr={remapped_metrics['hard_fpr']:.4f}"
    )

    # Holdout eval with feature path through ensemble.
    if len(test_idx):
        test_raw = np.asarray(
            ensemble.predict_chunk_scores(
                [chunks[i] for i in test_idx],
                x_all[test_idx],
            ),
            dtype=float,
        )
        test_remapped = _remap(test_raw)
        test_reward, test_metrics = reward(test_remapped, y[test_idx])
        print(
            f"HOLDOUT remapped reward={test_reward:.4f} "
            f"ap={test_metrics['ap_score']:.4f} "
            f"hard_bot_recall={test_metrics['hard_bot_recall']:.4f} "
            f"hard_fpr={test_metrics['hard_fpr']:.4f}"
        )

    metadata: dict[str, Any] = {
        "model_name": "poker44-hybrid-v1",
        "model_version": (
            "1.2.0-safety" if args.enable_safety_budget else "1.1.0-logic"
        ),
        "framework": "hybrid:logic-tracks+ExtraTrees+RF+HistGBM(+LGBM)+LogisticMeta",
        "benchmark_rows": int(len(y)),
        "feature_count": int(len(feature_names)),
        "logic_feature_count": int(sum(1 for n in feature_names if n.startswith("logic_"))),
        "score_remap": score_remap,
        "logic_blend_enabled": True,
        "logic_blend_max": 0.35,
        "train_dates": sorted(set(dates[i] for i in train_idx)),
        "holdout_dates": sorted(set(dates[i] for i in test_idx)),
        "human_weight": float(args.human_weight),
        "robust_features_only": (not args.no_robust_filter),
        "gpu_required": False,
        "notes": (
            "Hybrid of cold n-gram/stack/remap + handcrafted logic tracks "
            "(bot-family priors) as ML features + soft logic blend. CPU-only."
            + (
                " Includes topk batch FPR safety budget for live overconfident batches."
                if args.enable_safety_budget
                else ""
            )
        ),
    }
    if args.enable_safety_budget:
        # Cap positives per request so live OOD batches cannot all land >=0.5
        # (which zeros reward via threshold_sanity / hard_fpr).
        metadata["batch_safety_budget"] = {
            "kind": "topk_v1",
            "max_positive_count": 9999,
            "max_positive_fraction": float(args.safety_fraction),
            "positive_floor": 0.55,
            "positive_ceiling": 0.95,
            "negative_ceiling": 0.45,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models": [ensemble],
        "model_weights": [1.0],
        "feature_names": feature_names,
        "metadata": metadata,
        "calibrator": None,
    }
    joblib.dump(artifact, out_path)
    print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
