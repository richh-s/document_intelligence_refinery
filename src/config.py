"""Centralised configuration for the triage layer.

All magic numbers live here.  Values can be overridden via environment
variables thanks to pydantic-settings BaseSettings.
"""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings


class TriageConfig(BaseSettings):
    """Configurable thresholds, weights, and ceilings for triage detectors."""

    # ── Sampling ──────────────────────────────────────────────────────
    SAMPLE_PAGES: int = 5

    # ── Origin thresholds ─────────────────────────────────────────────
    INK_DENSITY_DIGITAL_MIN: float = 0.02
    WHITESPACE_RATIO_THRESHOLD: float = 0.95
    VECTOR_COUNT_SCANNED_MAX: int = 5
    ORIGIN_DIGITAL_THRESHOLD: float = 0.6
    ORIGIN_SCANNED_THRESHOLD: float = 0.4

    # ── Origin signal weights (must sum to 1.0) ──────────────────────
    ORIGIN_WEIGHTS: dict[str, float] = {
        "ink_density": 0.35,
        "whitespace_ratio": 0.25,
        "image_ratio": 0.20,
        "font_presence": 0.10,
        "vector_count": 0.10,
    }

    # ── Layout normalization ceilings ─────────────────────────────────
    LAYOUT_FONT_CEILING: float = 10.0
    LAYOUT_GRAPHIC_GROUP_CEILING: float = 20.0
    LAYOUT_LINE_VAR_CEILING: float = 200.0
    LAYOUT_COLUMN_CEILING: float = 3.0
    LAYOUT_TABLE_CEILING: float = 5.0

    # ── Vector clustering ─────────────────────────────────────────────
    VECTOR_CLUSTER_DISTANCE: float = 5.0  # px

    # ── Layout signal weights (must sum to 1.0) ──────────────────────
    LAYOUT_WEIGHTS: dict[str, float] = {
        "font": 0.2,
        "graphic_group": 0.2,
        "line_var": 0.2,
        "column": 0.2,
        "table": 0.2,
    }

    # ── Layout boundaries ─────────────────────────────────────────────
    LAYOUT_TABLE_HEAVY_THRESHOLD: float = 0.4
    LAYOUT_FIGURE_HEAVY_THRESHOLD: float = 0.5
    LAYOUT_MULTI_COLUMN_THRESHOLD: float = 0.5
    LAYOUT_MIXED_ACTIVE_SIGNALS: int = 3

    # ── Language ──────────────────────────────────────────────────────
    LANGUAGE_MIN_TOKENS: int = 20

    # ── Domain ────────────────────────────────────────────────────────
    DOMAIN_MIN_TOKENS: int = 50

    # ── Column detection ──────────────────────────────────────────────
    COLUMN_GAP_THRESHOLD: float = 20.0  # px gap between word clusters

    # ── Validators ────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_weights(self) -> Self:
        for name in ("ORIGIN_WEIGHTS", "LAYOUT_WEIGHTS"):
            weights = getattr(self, name)
            total = sum(weights.values())
            if abs(total - 1.0) >= 1e-6:
                msg = f"{name} must sum to 1.0, got {total}"
                raise ValueError(msg)
        return self

    model_config = {"env_prefix": "TRIAGE_"}
