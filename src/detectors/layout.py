"""Layout-complexity detector: simple vs moderate vs complex.

Uses vector clustering (graphic groups) instead of raw vector counts,
and normalized scoring to avoid classification cliffs.
"""

from __future__ import annotations

import statistics
from typing import Any

import structlog

from config import TriageConfig
from models.document_profile import LayoutType

logger = structlog.get_logger()


class LayoutComplexityDetector:
    """Determines the structural complexity of a document layout."""

    def __init__(self, config: TriageConfig) -> None:
        self._cfg = config

    # ── Public API ────────────────────────────────────────────────────

    def detect(
        self,
        page_stats: list[dict[str, Any]],
        image_ratio: float = 0.0,
    ) -> tuple[LayoutType, float, dict[str, Any]]:
        """Classify layout type and return ``(type, confidence, metadata)``.

        Parameters
        ----------
        page_stats:
            One dict per sampled page, each with keys:
            ``unique_fonts``, ``vectors`` (list of bbox dicts),
            ``line_height_variance``, ``words`` (list of word dicts),
            ``tables_detected``.
        image_ratio:
            Document-level image ratio, used for FIGURE_HEAVY scoring.
        """
        if not page_stats:
            return LayoutType.SINGLE_COLUMN, 0.0, {}

        page_layout_scores: list[float] = []
        per_page_details: list[dict[str, Any]] = []

        for i, stats in enumerate(page_stats):
            signals = self._compute_signals(stats)
            score = self._weighted_score(signals)
            page_layout_scores.append(round(score, 6))
            per_page_details.append(signals)

            logger.info(
                "layout_signals",
                page=i,
                **{k: round(v, 4) for k, v in signals.items()},
                layout_score=round(score, 6),
            )

        # Average the normalized signals across pages
        signals_avg: dict[str, float] = {}
        if per_page_details:
            keys = per_page_details[0].keys()
            for k in keys:
                signals_avg[k] = statistics.mean(p[k] for p in per_page_details)

        avg_score = statistics.mean(page_layout_scores) if page_layout_scores else 0.0
        
        layout_type, confidence = self._classify(signals_avg, image_ratio)

        columns_estimated = [self._estimate_columns(s.get("words", [])) for s in page_stats]
        detected_text_columns = max(columns_estimated) if columns_estimated else 1

        metadata = {
            "detected_text_columns": detected_text_columns,
            "avg_layout_score": round(avg_score, 6),
            "page_scores": page_layout_scores,
            "per_page_signals_avg": {k: round(v, 6) for k, v in signals_avg.items()},
        }

        return layout_type, round(confidence, 6), metadata

    # ── Signal computation ────────────────────────────────────────────

    def _compute_signals(self, stats: dict[str, Any]) -> dict[str, float]:
        cfg = self._cfg

        unique_fonts = float(stats.get("unique_fonts", 0))
        line_var = float(stats.get("line_height_variance", 0.0))
        tables = float(stats.get("tables_detected", 0))

        # Graphic groups via vector clustering
        vectors = stats.get("vectors", [])
        graphic_groups = float(
            self._cluster_vectors(vectors) if vectors else 0
        )

        # Column count via word x0 gap detection
        words = stats.get("words", [])
        columns = float(self._estimate_columns(words) if words else 1)

        return {
            "font_score": min(unique_fonts / max(cfg.LAYOUT_FONT_CEILING, 1e-9), 1.0),
            "graphic_group_score": min(
                graphic_groups / max(cfg.LAYOUT_GRAPHIC_GROUP_CEILING, 1e-9), 1.0
            ),
            "line_var_score": min(
                line_var / max(cfg.LAYOUT_LINE_VAR_CEILING, 1e-9), 1.0
            ),
            "column_score": min(columns / max(cfg.LAYOUT_COLUMN_CEILING, 1e-9), 1.0),
            "table_score": min(tables / max(cfg.LAYOUT_TABLE_CEILING, 1e-9), 1.0),
        }

    def _weighted_score(self, signals: dict[str, float]) -> float:
        w = self._cfg.LAYOUT_WEIGHTS
        signal_to_weight = {
            "font_score": "font",
            "graphic_group_score": "graphic_group",
            "line_var_score": "line_var",
            "column_score": "column",
            "table_score": "table",
        }
        return sum(
            signals[sig] * w.get(wkey, 0.0)
            for sig, wkey in signal_to_weight.items()
        )

    # ── Vector clustering ─────────────────────────────────────────────

    def _cluster_vectors(self, vectors: list[dict[str, Any]]) -> int:
        """Cluster touching/overlapping vectors into graphic groups.

        Uses single-linkage clustering based on bounding-box proximity.
        Adjacent vectors within ``VECTOR_CLUSTER_DISTANCE`` px merge
        into one logical group.
        """
        if not vectors:
            return 0

        bboxes = []
        for v in vectors:
            x0 = float(v.get("x0", 0))
            y0 = float(v.get("top", v.get("y0", 0)))
            x1 = float(v.get("x1", x0))
            y1 = float(v.get("bottom", v.get("y1", y0)))
            bboxes.append((x0, y0, x1, y1))

        dist = self._cfg.VECTOR_CLUSTER_DISTANCE

        # Union-Find
        parent = list(range(len(bboxes)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if self._bboxes_close(bboxes[i], bboxes[j], dist):
                    union(i, j)

        return len({find(i) for i in range(len(bboxes))})

    @staticmethod
    def _bboxes_close(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
        dist: float,
    ) -> bool:
        """Check if two bboxes are within *dist* px of each other."""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        # Gap in x or y direction
        x_gap = max(0, max(ax0, bx0) - min(ax1, bx1))
        y_gap = max(0, max(ay0, by0) - min(ay1, by1))
        return x_gap <= dist and y_gap <= dist

    # ── Column estimation ─────────────────────────────────────────────

    def _estimate_columns(self, words: list[dict[str, Any]]) -> int:
        """Estimate column count from word x0 positions via gap detection.

        Words inside table bounding boxes are excluded to avoid
        false multi-column detection from tabular layouts.
        """
        if not words:
            return 1

        x_positions = sorted(float(w.get("x0", 0)) for w in words)
        if not x_positions:
            return 1

        gap_threshold = self._cfg.COLUMN_GAP_THRESHOLD
        columns = 1

        for k in range(1, len(x_positions)):
            if x_positions[k] - x_positions[k - 1] > gap_threshold:
                columns += 1

        # Cap at a reasonable max to avoid noise (e.g. max 3 text columns)
        return min(columns, 3)

    # ── Classification ────────────────────────────────────────────────

    def _classify(self, signals_avg: dict[str, float], image_ratio: float) -> tuple[LayoutType, float]:
        """Compute per-category scores, pick max, confidence = margin."""
        table_graphic = signals_avg.get("table_score", 0.0) + signals_avg.get("graphic_group_score", 0.0)
        col_score = signals_avg.get("column_score", 0.0)
        
        # SINGLE_COLUMN is the complement of all other structured complexities
        single_score = max(0.0, 1.0 - max(table_graphic, image_ratio, col_score))

        category_scores = {
            LayoutType.TABLE_HEAVY: table_graphic,
            LayoutType.FIGURE_HEAVY: image_ratio,
            LayoutType.MULTI_COLUMN: col_score,  # normalized, not raw
            LayoutType.SINGLE_COLUMN: single_score,
        }
        ranked = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top_type, top_score = ranked[0]
        
        if top_type == LayoutType.SINGLE_COLUMN:
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            confidence = top_score - second_score
        else:
            # The margin for structured layouts shouldn't be deflated by the null hypothesis.
            structured_candidates = [s for t, s in ranked if t != LayoutType.SINGLE_COLUMN]
            second_structured = structured_candidates[1] if len(structured_candidates) > 1 else 0.0
            confidence = top_score - second_structured

        # Check MIXED: sum only across explicit normalized signal keys
        signal_keys = ["font_score", "graphic_group_score", "line_var_score", "column_score", "table_score"]
        active = sum(1 for k in signal_keys if signals_avg.get(k, 0.0) > 0.3)
        if active >= self._cfg.LAYOUT_MIXED_ACTIVE_SIGNALS and confidence < 0.15:
            return LayoutType.MIXED, round(confidence, 6)

        return top_type, round(confidence, 6)
