"""Hybrid Poker44 model loader: cold remap + optional mild hot safety budget."""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from poker44_ml.features import chunk_features
from poker44_ml.logic_tracks import blend_ml_with_logic, chunk_logic_tracks

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


class Poker44Model:
    """Runtime wrapper for hybrid stacked artifacts."""

    def __init__(self, model_path: str | Path):
        if joblib is None:
            raise RuntimeError("joblib is required to load Poker44 models.")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        artifact = joblib.load(self.model_path)
        self.models = list(artifact.get("models") or [])
        if not self.models and artifact.get("model") is not None:
            self.models = [artifact["model"]]
        if not self.models:
            raise RuntimeError("Model artifact contains no models.")

        self.feature_names = list(artifact.get("feature_names") or [])
        self.metadata = dict(artifact.get("metadata") or {})
        self.calibrator = artifact.get("calibrator")
        self.score_logit_bias = float(self.metadata.get("score_logit_bias", 0.0) or 0.0)
        self.score_logit_temperature = max(
            float(self.metadata.get("score_logit_temperature", 1.0) or 1.0),
            1e-6,
        )
        score_remap = self.metadata.get("score_remap")
        if isinstance(score_remap, dict) and score_remap.get("kind"):
            self.score_remap: dict[str, Any] = score_remap
        else:
            self.score_remap = {}
        self.model_weights = list(
            artifact.get("model_weights")
            or self.metadata.get("model_weights")
            or [1.0 for _ in self.models]
        )
        # Mild hot-style safety budget. Default OFF unless stamped into artifact
        # or enabled via metadata.kind == topk_v1.
        self.batch_safety_budget = self.metadata.get("batch_safety_budget")
        # Soft logic gate: blend ML with rule prior when confidence is high.
        # Default ON lightly; disable with POKER44_LOGIC_BLEND=0.
        env_blend = os.getenv("POKER44_LOGIC_BLEND", "").strip().lower()
        if env_blend in {"0", "false", "off", "no"}:
            self.logic_blend_enabled = False
            self.logic_blend_max = 0.0
        elif env_blend in {"1", "true", "on", "yes"}:
            self.logic_blend_enabled = True
            self.logic_blend_max = float(os.getenv("POKER44_LOGIC_BLEND_MAX", "0.35"))
        else:
            self.logic_blend_enabled = bool(
                self.metadata.get("logic_blend_enabled", True)
            )
            self.logic_blend_max = float(
                self.metadata.get(
                    "logic_blend_max",
                    float(os.getenv("POKER44_LOGIC_BLEND_MAX", "0.35")),
                )
            )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    def _aligned_rows(self, chunks: list[list[dict[str, Any]]]) -> list[list[float]]:
        rows: list[list[float]] = []
        for chunk in chunks:
            features = chunk_features(chunk)
            features["hand_count"] = float(len(chunk))
            if not self.feature_names:
                self.feature_names = sorted(features)
            rows.append([float(features.get(name, 0.0)) for name in self.feature_names])
        return rows

    def _raw_model_scores(
        self,
        rows: list[list[float]],
        chunks: list[list[dict[str, Any]]] | None = None,
    ) -> list[float]:
        if not rows:
            return []
        matrix = np.asarray(rows, dtype=np.float64)
        columns: list[np.ndarray] = []
        weights: list[float] = []
        for model, weight in zip(self.models, self.model_weights):
            w = float(weight)
            if w <= 0.0:
                continue
            if (
                chunks is not None
                and hasattr(model, "predict_chunk_scores")
                and not isinstance(model, type(self))
            ):
                try:
                    col = np.asarray(
                        model.predict_chunk_scores(chunks, feature_rows=rows),
                        dtype=float,
                    )
                except TypeError:
                    col = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
            elif hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(matrix))
                col = proba[:, 1] if proba.ndim == 2 else proba
            else:
                col = np.asarray(model.predict(matrix), dtype=float)
            columns.append(np.clip(np.asarray(col, dtype=float).ravel(), 0.0, 1.0))
            weights.append(w)

        if not columns:
            return [0.5 for _ in rows]
        stacked = np.stack(columns, axis=1)
        w = np.asarray(weights, dtype=float)
        w = w / max(float(w.sum()), 1e-12)
        blended = stacked @ w
        return [float(v) for v in blended]

    def _apply_calibrator(self, scores: list[float]) -> list[float]:
        if not scores or self.calibrator is None:
            return [self._clamp01(v) for v in scores]
        arr = np.asarray(scores, dtype=float)
        if hasattr(self.calibrator, "transform"):
            out = np.asarray(self.calibrator.transform(arr), dtype=float)
        elif hasattr(self.calibrator, "predict"):
            out = np.asarray(self.calibrator.predict(arr), dtype=float)
        elif hasattr(self.calibrator, "predict_proba"):
            out = np.asarray(self.calibrator.predict_proba(arr.reshape(-1, 1))[:, 1])
        else:
            out = arr
        return [self._clamp01(float(v)) for v in out]

    def _apply_score_remap(self, scores: list[float]) -> list[float]:
        if not scores or not self.score_remap:
            return [self._clamp01(v) for v in scores]
        if self.score_remap.get("kind") != "threshold_logit_v1":
            return [self._clamp01(v) for v in scores]
        threshold = float(self.score_remap.get("threshold", 0.5))
        temperature = max(float(self.score_remap.get("temperature", 0.25)), 1e-6)
        out: list[float] = []
        for score in scores:
            remapped = self._sigmoid((float(score) - threshold) / temperature)
            out.append(self._clamp01(remapped))
        return out

    def _apply_score_logit(self, scores: list[float]) -> list[float]:
        if abs(self.score_logit_bias) < 1e-12 and abs(self.score_logit_temperature - 1.0) < 1e-12:
            return [self._clamp01(v) for v in scores]
        out: list[float] = []
        for score in scores:
            value = max(1e-6, min(1.0 - 1e-6, float(score)))
            logit = math.log(value / (1.0 - value))
            adjusted = (logit + self.score_logit_bias) / self.score_logit_temperature
            out.append(self._clamp01(self._sigmoid(adjusted)))
        return out

    def _apply_batch_safety_budget(self, scores: list[float]) -> list[float]:
        """Mild hot-king topk reshape. Only runs if artifact enables it."""
        config = self.batch_safety_budget
        if not scores or not isinstance(config, dict):
            return [self._clamp01(v) for v in scores]
        if config.get("kind") != "topk_v1":
            return [self._clamp01(v) for v in scores]

        count = len(scores)
        max_positive_count = int(config.get("max_positive_count", 9999))
        max_positive_fraction = float(config.get("max_positive_fraction", 0.0) or 0.0)
        positive_floor = float(config.get("positive_floor", 0.501))
        positive_ceiling = float(config.get("positive_ceiling", 0.55))
        negative_ceiling = float(config.get("negative_ceiling", 0.49))

        if max_positive_fraction > 0.0:
            max_positive_count = min(
                max_positive_count,
                max(1, int(math.floor(count * max_positive_fraction))),
            )
        max_positive_count = max(0, min(count, max_positive_count))
        positive_floor = self._clamp01(positive_floor)
        positive_ceiling = self._clamp01(max(positive_floor, positive_ceiling))
        negative_ceiling = min(self._clamp01(negative_ceiling), positive_floor - 1e-6)

        ranked = sorted(
            [(i, self._clamp01(v)) for i, v in enumerate(scores)],
            key=lambda item: item[1],
            reverse=True,
        )
        # Only promote scores that already look bot-like after remap.
        eligible = [(i, s) for i, s in ranked if s >= 0.5]
        positives = eligible[:max_positive_count]
        positive_ids = {i for i, _ in positives}
        output = [0.0 for _ in scores]

        if positives:
            denom = max(1, len(positives) - 1)
            for rank, (index, _score) in enumerate(positives):
                relative = 1.0 - (rank / denom if denom else 0.0)
                output[index] = positive_floor + relative * (positive_ceiling - positive_floor)

        negatives = [(i, s) for i, s in ranked if i not in positive_ids]
        if negatives:
            vals = [s for _, s in negatives]
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 1e-9)
            for index, score in negatives:
                relative = (score - lo) / span
                output[index] = max(0.0, min(negative_ceiling, relative * negative_ceiling))

        return [round(self._clamp01(v), 6) for v in output]

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if not chunks:
            return []
        rows = self._aligned_rows(chunks)
        raw_scores = self._raw_model_scores(rows, chunks=chunks)
        calibrated_scores = self._apply_calibrator(raw_scores)
        remapped_scores = self._apply_score_remap(calibrated_scores)
        logit_scores = self._apply_score_logit(remapped_scores)
        if self.logic_blend_enabled and self.logic_blend_max > 0.0:
            blended: list[float] = []
            for chunk, score in zip(chunks, logit_scores):
                tracks = chunk_logic_tracks(chunk)
                blended.append(
                    blend_ml_with_logic(
                        score,
                        tracks.get("logic_rule_prior", 0.5),
                        tracks.get("logic_rule_confidence", 0.0),
                        max_blend=self.logic_blend_max,
                    )
                )
            logit_scores = blended
        budgeted_scores = self._apply_batch_safety_budget(logit_scores)
        return [round(self._clamp01(v), 6) for v in budgeted_scores]
