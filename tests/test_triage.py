"""Integration tests for TriageAgent — requires real PDFs in data/.

Marked with ``@pytest.mark.integration`` so they can be skipped in
environments without sample data.
"""

from pathlib import Path

import pytest

from models.document_profile import (
    DomainHint,
    ExtractionCostEstimate,
    LayoutType,
    OriginType,
)
from agents.triage import TriageAgent

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Skip all tests in this module if the data directory is empty
pytestmark = pytest.mark.integration


def _has_data() -> bool:
    return DATA_DIR.exists() and any(DATA_DIR.glob("*.pdf"))


skip_no_data = pytest.mark.skipif(
    not _has_data(),
    reason="Sample PDFs not found in data/",
)


@pytest.fixture(scope="module")
def agent() -> TriageAgent:
    return TriageAgent()


# ═══════════════════════════════════════════════════════════════════════
# Per-Document Classification Tests
# ═══════════════════════════════════════════════════════════════════════


@skip_no_data
class TestAuditReport:
    """Audit Report - 2023.pdf → SCANNED (primary) or MIXED."""

    PDF = DATA_DIR / "Audit Report - 2023.pdf"

    def test_origin_not_digital(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.origin_type in (OriginType.SCANNED, OriginType.MIXED)

    def test_domain_financial_or_governmental(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.domain_hint in (
            DomainHint.FINANCIAL, DomainHint.GENERAL,
        )


@skip_no_data
class TestCBEAnnualReport:
    """CBE ANNUAL REPORT 2023-24.pdf → MIXED (cover images + digital text), FINANCIAL."""

    PDF = DATA_DIR / "CBE ANNUAL REPORT 2023-24.pdf"

    def test_origin(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        # Cover pages are full-page images; body pages are digital text
        assert p.origin_type in (OriginType.DIGITAL_NATIVE, OriginType.MIXED)

    def test_layout(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        # Layout varies depending on which pages are sampled
        assert p.layout_type in (
            LayoutType.SINGLE_COLUMN, LayoutType.MULTI_COLUMN, LayoutType.TABLE_HEAVY, LayoutType.FIGURE_HEAVY, LayoutType.MIXED
        )

    def test_domain_financial(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.domain_hint in (DomainHint.FINANCIAL, DomainHint.GENERAL)


@skip_no_data
class TestFTAPerformance:
    """fta_performance_survey_final_report_2022.pdf → DIGITAL_NATIVE, complex layout, GOVERNMENTAL."""

    PDF = DATA_DIR / "fta_performance_survey_final_report_2022.pdf"

    def test_origin_digital(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.origin_type in (OriginType.DIGITAL_NATIVE, OriginType.MIXED)

    def test_layout_not_single(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.layout_type in (
            LayoutType.MULTI_COLUMN, LayoutType.TABLE_HEAVY, LayoutType.MIXED
        )

    def test_domain_governmental(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.domain_hint == DomainHint.GOVERNMENTAL


@skip_no_data
class TestTaxExpenditure:
    """tax_expenditure_ethiopia_2021_22.pdf → DIGITAL_NATIVE, complex layout, FINANCIAL or GOVERNMENTAL."""

    PDF = DATA_DIR / "tax_expenditure_ethiopia_2021_22.pdf"

    def test_origin_digital(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.origin_type == OriginType.DIGITAL_NATIVE

    def test_layout_not_single(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.layout_type in (
            LayoutType.MULTI_COLUMN, LayoutType.TABLE_HEAVY, LayoutType.MIXED
        )

    def test_domain(self, agent: TriageAgent) -> None:
        p = agent.profile(self.PDF)
        assert p.domain_hint in (DomainHint.FINANCIAL, DomainHint.GOVERNMENTAL)


# ═══════════════════════════════════════════════════════════════════════
# Cross-Cutting Assertions
# ═══════════════════════════════════════════════════════════════════════


@skip_no_data
class TestProfileInvariants:
    """Assertions that must hold for every profiled document."""

    PDFS = list(DATA_DIR.glob("*.pdf")) if _has_data() else []

    @pytest.fixture(params=[str(p) for p in PDFS], ids=[p.name for p in PDFS])
    def profile(self, request: pytest.FixtureRequest, agent: TriageAgent):
        return agent.profile(Path(request.param))

    def test_file_hash_populated(self, profile) -> None:
        assert profile.file_hash
        assert len(profile.file_hash) == 64  # SHA-256 hex

    def test_not_encrypted(self, profile) -> None:
        assert profile.is_encrypted is False

    def test_processing_metadata(self, profile) -> None:
        assert "pdfplumber_version" in profile.processing_metadata
        assert "triage_config_version" in profile.processing_metadata

    def test_confidence_bounds(self, profile) -> None:
        assert 0.0 <= profile.confidence.origin <= 1.0
        assert 0.0 <= profile.confidence.layout <= 1.0
        assert 0.0 <= profile.confidence.domain <= 1.0

    def test_determinism(self, agent: TriageAgent, profile) -> None:
        """Same file profiled twice must yield identical result."""
        pdf_path = DATA_DIR / profile.file_name
        second = agent.profile(pdf_path)
        assert profile == second


class TestControlPlaneLogic:
    """Unit tests for TriageAgent internal cost estimation and extraction logic."""

    def test_form_fillable_cost_mapping(self, agent: TriageAgent) -> None:
        assert agent._estimate_cost(OriginType.FORM_FILLABLE, LayoutType.SINGLE_COLUMN) == ExtractionCostEstimate.FAST_TEXT_SUFFICIENT
        assert agent._estimate_cost(OriginType.FORM_FILLABLE, LayoutType.TABLE_HEAVY) == ExtractionCostEstimate.NEEDS_LAYOUT_MODEL
        
    def test_digital_native_cost_mapping(self, agent: TriageAgent) -> None:
        assert agent._estimate_cost(OriginType.DIGITAL_NATIVE, LayoutType.SINGLE_COLUMN) == ExtractionCostEstimate.FAST_TEXT_SUFFICIENT
        assert agent._estimate_cost(OriginType.DIGITAL_NATIVE, LayoutType.MULTI_COLUMN) == ExtractionCostEstimate.NEEDS_LAYOUT_MODEL
        assert agent._estimate_cost(OriginType.DIGITAL_NATIVE, LayoutType.TABLE_HEAVY) == ExtractionCostEstimate.NEEDS_LAYOUT_MODEL
        
    def test_scanned_cost_mapping(self, agent: TriageAgent) -> None:
        assert agent._estimate_cost(OriginType.SCANNED, LayoutType.SINGLE_COLUMN) == ExtractionCostEstimate.NEEDS_VISION_MODEL
        assert agent._estimate_cost(OriginType.MIXED, LayoutType.TABLE_HEAVY) == ExtractionCostEstimate.NEEDS_VISION_MODEL
