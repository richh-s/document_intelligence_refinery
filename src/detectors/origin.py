"""Origin-type detector: digital-native vs scanned vs mixed.

Uses per-page ink-density analysis instead of raw character counts
to correctly handle sparse digital pages (cover pages, title pages).
"""

from __future__ import annotations

from typing import Any

import structlog

from config import TriageConfig
from models.document_profile import OriginType

logger = structlog.get_logger()


class OriginTypeDetector:
    """Determines whether a PDF is digital-native, scanned, or mixed.

    Scores each sampled page independently, then aggregates.
    """

    def __init__(self, config: TriageConfig) -> None:
        self._cfg = config

    # ── Public API ────────────────────────────────────────────────────

    def detect(
        self,
        page_stats: list[dict[str, Any]],
    ) -> tuple[OriginType, float, dict[str, Any]]:
        """Classify origin and return ``(origin_type, confidence, metadata)``.

        Parameters
        ----------
        page_stats:
            One dict per sampled page, each containing keys:
            ``ink_density``, ``char_density``, ``whitespace_ratio``, ``image_ratio``,
            ``font_count``, ``vector_count``.
        """
        if not page_stats:
            return OriginType.MIXED, 0.0, {"page_scores": [], "page_labels": []}

        page_scores: list[float] = []
        page_labels: list[str] = []

        for i, stats in enumerate(page_stats):
            score = self._score_page(stats)
            label = self._classify_page(score)
            page_scores.append(round(score, 6))
            page_labels.append(label)

            logger.info(
                "origin_signals",
                page=i,
                ink_density=stats.get("ink_density"),
                char_density=stats.get("char_density"),
                whitespace_ratio=stats.get("whitespace_ratio"),
                image_ratio=stats.get("image_ratio"),
                font_count=stats.get("font_count"),
                vector_count=stats.get("vector_count"),
                digital_score=round(score, 6),
                label=label,
            )

        origin_type = self._aggregate(page_labels)
        confidence = self._compute_confidence(page_scores, origin_type)

        metadata = {
            "page_scores": page_scores,
            "page_labels": page_labels,
            "char_density": sum(s.get("char_density", 0.0) for s in page_stats) / len(page_stats),
            "whitespace_ratio": sum(s.get("whitespace_ratio", 0.0) for s in page_stats) / len(page_stats),
            "ink_density": sum(s.get("ink_density", 0.0) for s in page_stats) / len(page_stats),
            "aggregation_method": "mean",
        }

        return origin_type, round(confidence, 6), metadata

    # ── Internals ─────────────────────────────────────────────────────

    def _score_page(self, stats: dict[str, Any]) -> float:
        """Compute a per-page ``digital_score ∈ [0, 1]``.

        Higher = more likely digital-native.
        """
        w = self._cfg.ORIGIN_WEIGHTS

        # ink_density signal: digital pages typically have 2-15% ink coverage
        ink = float(stats.get("ink_density", 0.0))
        ink_min = self._cfg.INK_DENSITY_DIGITAL_MIN
        # Smooth ramp: anything above ink_min scores well
        ink_signal = min(ink / max(ink_min, 1e-9), 1.0)
        
        # explicit char_density signal: > 0 characters per unit area supports digital/mixed
        char_density = float(stats.get("char_density", 0.0))
        char_signal = 1.0 if char_density > 0.001 else 0.0

        # whitespace_ratio signal: scanned pages → near 1.0
        ws = float(stats.get("whitespace_ratio", 1.0))
        ws_threshold = self._cfg.WHITESPACE_RATIO_THRESHOLD
        ws_signal = 1.0 - min(ws / ws_threshold, 1.0) if ws_threshold > 0 else 0.0

        # image_ratio signal: scanned pages are dominated by images
        img = float(stats.get("image_ratio", 0.0))
        img_signal = 1.0 - min(img, 1.0)

        # font_presence signal: digital text has fonts; scans don't
        font_count = int(stats.get("font_count", 0))
        font_signal = min(font_count / 1.0, 1.0)  # ≥1 font → 1.0

        # vector_count signal: digital docs tend to have more vectors
        vectors = int(stats.get("vector_count", 0))
        vec_max = self._cfg.VECTOR_COUNT_SCANNED_MAX
        vec_signal = min(vectors / max(vec_max, 1), 1.0)

        score = (
            w.get("ink_density", 0.0) * ink_signal
            + w.get("char_density", 0.0) * char_signal
            + w.get("whitespace_ratio", 0.0) * ws_signal
            + w.get("image_ratio", 0.0) * img_signal
            + w.get("font_presence", 0.0) * font_signal
            + w.get("vector_count", 0.0) * vec_signal
        )
        return score

    def _classify_page(self, score: float) -> str:
        if score >= self._cfg.ORIGIN_DIGITAL_THRESHOLD:
            return "digital"
        if score <= self._cfg.ORIGIN_SCANNED_THRESHOLD:
            return "scanned"
        return "ambiguous"

    def _aggregate(self, labels: list[str]) -> OriginType:
        digital_count = labels.count("digital")
        scanned_count = labels.count("scanned")
        # ambiguous pages are neutral — don't trigger MIXED

        if digital_count > 0 and scanned_count == 0:
            return OriginType.DIGITAL_NATIVE
        if scanned_count > 0 and digital_count == 0:
            return OriginType.SCANNED
        if digital_count > 0 and scanned_count > 0:
            return OriginType.MIXED

        # All pages ambiguous → use average score direction
        return OriginType.DIGITAL_NATIVE

    @staticmethod
    def _compute_confidence(scores: list[float], origin_type: OriginType) -> float:
        """Origin confidence derived deterministically from mean page_scores."""
        if not scores:
            return 0.0
        mean_score = sum(scores) / len(scores)
        if origin_type == OriginType.SCANNED:
            return 1.0 - mean_score
        if origin_type == OriginType.MIXED:
            return 1.0 - abs(mean_score - 0.5) * 2.0
        return mean_score
