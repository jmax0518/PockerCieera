"""Dump model risk_scores next to public ground-truth labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from poker44.score.scoring import reward  # noqa: E402
from poker44_ml.inference import Poker44Model  # noqa: E402
from training.train_hybrid import load_benchmark_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "hybrid_v1.joblib"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "benchmark"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "evals"),
    )
    parser.add_argument("--tag", default="compare")
    args = parser.parse_args()

    chunks, labels, dates = load_benchmark_rows(Path(args.data_dir))
    y = np.asarray(labels, dtype=int)
    model = Poker44Model(args.model)
    scores = np.asarray(model.predict_chunk_scores(chunks), dtype=float)
    rew, metrics = reward(scores, y)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (score, label, date) in enumerate(zip(scores, y, dates)):
        rows.append(
            {
                "idx": i,
                "source_date": date,
                "y_true": int(label),
                "y_true_label": "bot" if label == 1 else "human",
                "risk_score": round(float(score), 6),
                "prediction_ge_0_5": bool(score >= 0.5),
                "hand_count": len(chunks[i]),
            }
        )

    csv_path = out_dir / f"{args.tag}_scores.csv"
    json_path = out_dir / f"{args.tag}_scores.json"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": str(args.model),
        "n": int(len(y)),
        "bots": int(y.sum()),
        "humans": int((y == 0).sum()),
        "pos_rate": float(np.mean(scores >= 0.5)),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_p50": float(np.median(scores)),
        "score_max": float(scores.max()),
        "human_pos_rate": float(np.mean(scores[y == 0] >= 0.5)) if (y == 0).any() else None,
        "bot_pos_rate": float(np.mean(scores[y == 1] >= 0.5)) if (y == 1).any() else None,
        "human_mean": float(scores[y == 0].mean()) if (y == 0).any() else None,
        "bot_mean": float(scores[y == 1].mean()) if (y == 1).any() else None,
        "reward": float(rew),
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()},
        "logic_blend_enabled": model.logic_blend_enabled,
        "batch_safety_budget": model.batch_safety_budget,
        "score_remap": model.score_remap,
        "head_20": rows[:20],
    }
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
