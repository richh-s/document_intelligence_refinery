"""Core data models for the Document Intelligence Refinery triage layer.

Defines the DocumentProfile schema and all supporting types used by
the TriageAgent and its detectors.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


# ── Classification Enums ──────────────────────────────────────────────


class OriginType(str, Enum):
    """How the PDF was produced."""

    DIGITAL_NATIVE = "digital_native"
    SCANNED = "scanned"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutType(str, Enum):
    """Structural layout category of the document."""

    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"

class ExtractionCostEstimate(str, Enum):
    """Estimated effort required to extract structure and text."""

    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"


class DomainHint(str, Enum):
    """Subject-matter domain of the document."""

    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GOVERNMENTAL = "governmental"
    GENERAL = "general"


# ── Confidence Model ──────────────────────────────────────────────────


class ConfidenceScores(BaseModel):
    """Per-dimension confidence scores.

    All values are rounded to 6 decimal places at construction
    to guarantee floating-point determinism across runs.
    """

    origin: float = Field(ge=0.0, le=1.0)
    layout: float = Field(ge=0.0, le=1.0)
    domain: float = Field(ge=0.0, le=1.0)
    language: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _round_values(self) -> Self:
        for field_name in ("origin", "layout", "domain", "language"):
            object.__setattr__(
                self, field_name, round(getattr(self, field_name), 6)
            )
        return self


# ── Document Profile ──────────────────────────────────────────────────


class DocumentProfile(BaseModel):
    """Complete classification output for a single PDF document.

    Assembled by the TriageAgent after running all detectors.
    """

    file_name: str
    doc_id: str
    file_hash: str  # SHA-256 hex digest
    origin_type: OriginType
    layout_type: LayoutType
    domain_hint: DomainHint
    language: str = "unknown"
    language_confidence: float = 0.0
    extraction_cost: ExtractionCostEstimate
    confidence: ConfidenceScores
    page_count: int = Field(ge=0)
    is_encrypted: bool = False
    warnings: list[str] = Field(default_factory=list)
    processing_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
