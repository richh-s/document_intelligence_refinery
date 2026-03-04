"""Unit tests for models.py — Pydantic schema validation."""

import pytest

from models.document_profile import (
    ConfidenceScores,
    DocumentProfile,
    DomainHint,
    ExtractionCostEstimate,
    LayoutType,
    OriginType,
)


class TestConfidenceScores:
    """ConfidenceScores construction, bounds, and FP rounding."""

    def test_valid_construction(self) -> None:
        cs = ConfidenceScores(origin=0.85, layout=0.5, domain=0.3, language=1.0)
        assert cs.origin == 0.85
        assert cs.layout == 0.5
        assert cs.domain == 0.3
        assert cs.language == 1.0

    def test_rounding_determinism(self) -> None:
        """Floating-point values are rounded to 6 decimals."""
        cs = ConfidenceScores(
            origin=0.123456789, layout=0.999999999, domain=0.0000001, language=0.5123456
        )
        assert cs.origin == 0.123457
        assert cs.layout == 1.0
        assert cs.domain == 0.0
        assert cs.language == 0.512346

    def test_bounds_rejection_above_one(self) -> None:
        with pytest.raises(Exception):
            ConfidenceScores(origin=1.5, layout=0.5, domain=0.5, language=0.5)

    def test_bounds_rejection_below_zero(self) -> None:
        with pytest.raises(Exception):
            ConfidenceScores(origin=-0.1, layout=0.5, domain=0.5, language=0.5)


class TestDocumentProfile:
    """DocumentProfile construction and serialisation."""

    def _make_profile(self, **overrides: object) -> DocumentProfile:
        defaults = {
            "file_name": "test.pdf",
            "doc_id": "abc123",
            "file_hash": "abc123",
            "origin_type": OriginType.DIGITAL_NATIVE,
            "layout_type": LayoutType.SINGLE_COLUMN,
            "domain_hint": DomainHint.GENERAL,
            "language": "en",
            "language_confidence": 1.0,
            "extraction_cost": ExtractionCostEstimate.FAST_TEXT_SUFFICIENT,
            "confidence": ConfidenceScores(origin=0.8, layout=0.5, domain=0.3, language=1.0),
            "page_count": 10,
            "is_encrypted": False,
            "warnings": [],
            "processing_metadata": {"triage_config_version": "1.0.0"},
            "metadata": {},
        }
        defaults.update(overrides)  # type: ignore[arg-type]
        return DocumentProfile(**defaults)  # type: ignore[arg-type]

    def test_construction(self) -> None:
        p = self._make_profile()
        assert p.file_name == "test.pdf"
        assert p.origin_type == OriginType.DIGITAL_NATIVE
        assert p.is_encrypted is False
        assert p.warnings == []

    def test_warnings_default_factory(self) -> None:
        """Two profiles must not share the same warnings list."""
        p1 = self._make_profile()
        p2 = self._make_profile()
        p1.warnings.append("test warning")
        assert "test warning" not in p2.warnings

    def test_serialisation_roundtrip(self) -> None:
        p = self._make_profile()
        data = p.model_dump()
        p2 = DocumentProfile(**data)
        assert p == p2

    def test_enum_values(self) -> None:
        assert OriginType.SCANNED.value == "scanned"
        assert LayoutType.TABLE_HEAVY.value == "table_heavy"
        assert DomainHint.FINANCIAL.value == "financial"
        assert ExtractionCostEstimate.NEEDS_VISION_MODEL.value == "needs_vision_model"
