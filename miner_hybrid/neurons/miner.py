"""Hybrid Poker44 miner: cold features/stack/remap + optional hot safety budget."""

import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import bittensor as bt

HYBRID_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HYBRID_ROOT.parent
sys.path.insert(0, str(HYBRID_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

try:
    from poker44_ml.inference import Poker44Model
    from poker44_ml.logic_tracks import chunk_logic_tracks
except ImportError:  # pragma: no cover
    Poker44Model = None
    chunk_logic_tracks = None


class Miner(BaseMinerNeuron):
    """Serve hybrid bot-risk scores over DetectionSynapse."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.max_hands_per_chunk_eval = max(
            0, int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK_EVAL", "120"))
        )
        self.model_path = Path(
            os.getenv(
                "POKER44_MODEL_PATH",
                str(HYBRID_ROOT / "models" / "hybrid_v1.joblib"),
            )
        )
        self.predictor = None
        self.backend = "heuristic"
        if Poker44Model is not None and self.model_path.exists():
            try:
                self.predictor = Poker44Model(self.model_path)
                self.backend = "hybrid-supervised"
            except Exception as err:
                bt.logging.warning(
                    f"Failed to load hybrid model at {self.model_path}: {err}. "
                    "Falling back to heuristic."
                )

        model_metadata = dict(self.predictor.metadata) if self.predictor is not None else {}
        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[
                Path(__file__).resolve(),
                HYBRID_ROOT / "poker44_ml" / "features.py",
                HYBRID_ROOT / "poker44_ml" / "logic_tracks.py",
                HYBRID_ROOT / "poker44_ml" / "inference.py",
                HYBRID_ROOT / "poker44_ml" / "stacked.py",
            ],
            defaults={
                "model_name": model_metadata.get("model_name") or "poker44-hybrid-v1",
                "model_version": model_metadata.get("model_version") or "1.2.0-safety",
                "framework": model_metadata.get("framework")
                or "hybrid-logic+tree-stack-cpu",
                "license": "MIT",
                "open_source": True,
                "repo_url": "https://github.com/jmax0518/PockerCieera",
                "repo_commit": os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip(),
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on Poker44 public benchmark only "
                    "(https://api.poker44.net/api/v1/benchmark). "
                    "No validator-private labels."
                ),
                "training_data_sources": ["released_training_benchmark"],
                "private_data_attestation": (
                    "No validator-private evaluation data used for training."
                ),
                "notes": (
                    "Hybrid miner: handcrafted logic tracks (bot-family priors) + "
                    "cold n-grams + stacked trees + threshold_logit remap + soft "
                    "logic/ML blend + batch FPR safety budget. CPU-trained. "
                    "Served by hotkey cierra-poker (UID 33)."
                ),
                "artifact_url": (
                    "https://github.com/jmax0518/PockerCieera/blob/main/"
                    "miner_hybrid/models/hybrid_v1.joblib"
                ),
                "model_card_url": "https://github.com/jmax0518/PockerCieera",
            },
        )
        if self.model_path.exists():
            self.model_manifest["artifact_sha256"] = self._sha256(self.model_path)
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"Hybrid miner started backend={self.backend} "
            f"model={self.model_path} "
            f"transparency={self.manifest_compliance.get('status')}"
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _compress_chunk(self, chunk: list) -> list:
        limit = self.max_hands_per_chunk_eval
        if limit <= 0 or len(chunk) <= limit:
            return chunk
        if limit == 1:
            return [chunk[0]]
        step = (len(chunk) - 1) / (limit - 1)
        indices = sorted({int(round(i * step)) for i in range(limit)})
        return [chunk[i] for i in indices]

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful = max(
            1,
            sum(action_counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")),
        )
        call_ratio = action_counts.get("call", 0) / meaningful
        check_ratio = action_counts.get("check", 0) / meaningful
        fold_ratio = action_counts.get("fold", 0) / meaningful
        raise_ratio = action_counts.get("raise", 0) / meaningful
        street_depth = len(hand.get("streets") or []) / 3.0
        showdown = 1.0 if (hand.get("outcome") or {}).get("showdown") else 0.0
        score = (
            0.32 * street_depth
            + 0.22 * showdown
            + 0.18 * cls._clamp01(call_ratio / 0.35)
            + 0.12 * cls._clamp01(check_ratio / 0.30)
            - 0.18 * cls._clamp01(fold_ratio / 0.55)
            - 0.10 * cls._clamp01(raise_ratio / 0.20)
        )
        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list) -> float:
        if not chunk:
            return 0.5
        # Prefer fused logic prior when ML artifact is unavailable.
        if chunk_logic_tracks is not None:
            tracks = chunk_logic_tracks(chunk)
            prior = float(tracks.get("logic_rule_prior", 0.5))
            heuristic = cls._clamp01(sum(cls._score_hand(h) for h in chunk) / len(chunk))
            # Blend classic heuristic with bot-family prior.
            return cls._clamp01(0.45 * heuristic + 0.55 * prior)
        return cls._clamp01(sum(cls._score_hand(h) for h in chunk) / len(chunk))

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = [self._compress_chunk(list(chunk or [])) for chunk in (synapse.chunks or [])]
        if self.predictor is not None:
            scores = self.predictor.predict_chunk_scores(chunks)
        else:
            scores = [self.score_chunk(chunk) for chunk in chunks]
        if len(scores) < len(chunks):
            scores = list(scores) + [0.0] * (len(chunks) - len(scores))
        elif len(scores) > len(chunks):
            scores = list(scores)[: len(chunks)]
        scores = [self._clamp01(score) for score in scores]
        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)
        if scores:
            smin = min(scores)
            smax = max(scores)
            smean = sum(scores) / len(scores)
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            sp50 = (
                sorted_scores[mid]
                if len(sorted_scores) % 2 == 1
                else 0.5 * (sorted_scores[mid - 1] + sorted_scores[mid])
            )
        else:
            smin = smax = smean = sp50 = 0.0
        bt.logging.info(
            f"Scored {len(chunks)} chunks backend={self.backend} "
            f"pos_rate={sum(synapse.predictions)/max(1,len(scores)):.3f} "
            f"min={smin:.3f} mean={smean:.3f} p50={sp50:.3f} max={smax:.3f} "
            f"head={[round(s, 3) for s in scores[:8]]}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner(config=Miner.config()) as miner:
        while True:
            bt.logging.info(f"Hybrid miner running | uid={miner.uid}")
            time.sleep(30)
