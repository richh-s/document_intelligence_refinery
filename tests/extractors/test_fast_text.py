"""Tests for the Strategy A Fast Text Extractor."""

import pytest
from pathlib import Path
from pypdf import PdfWriter

from models.document_profile import DocumentProfile, OriginType, LayoutType, ConfidenceScores
from strategies.fast_text import FastTextExtractor


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create a physically empty single-page PDF for testing sparse failures."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)  # US Letter size
    pdf_path = tmp_path / "empty_test.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def test_nonsense_ratio_checker():
    """Verify N-gram heuristic junk rejection logic."""
    extractor = FastTextExtractor({"NONSENSE_RATIO_MAX": 0.3})
    
    # Valid tokens
    assert not extractor._is_nonsense("Financial")
    assert not extractor._is_nonsense("Report")
    assert not extractor._is_nonsense("12,345.67")
    
    # Invalid / Junk OCR tokens
    assert extractor._is_nonsense("a" * 41)  # Too long
    assert extractor._is_nonsense("$$$$---11") # No alphabet, mostly symbols
    assert extractor._is_nonsense("ssssss")  # Repeated characters
    assert extractor._is_nonsense("xxxxxxx") # Repeated characters


def test_sparse_text_confidence_penalty(empty_pdf: Path):
    """Verify Strategy A heavily penalizes pages with near-zero character counts."""
    extractor = FastTextExtractor({"NONSENSE_RATIO_MAX": 0.3})
    profile = DocumentProfile(
        file_hash="dummy_hash",
        file_name="empty_test.pdf",
        doc_id="dummy_hash",
        file_path=str(empty_pdf),
        origin_type=OriginType.DIGITAL_NATIVE,
        layout_type=LayoutType.SINGLE_COLUMN,
        language="en",
        language_confidence=1.0,
        domain_hint="general",
        extraction_cost="fast_text_sufficient",
        confidence=ConfidenceScores(origin=1.0, layout=1.0, language=1.0, domain=1.0),
        page_count=1
    )
    
    result = extractor.extract(empty_pdf, profile)
    
    assert result.confidence < 0.2  # Should be severely penalized (e.g. 0.1)
    assert len(result.document.pages) == 1
    assert result.document.pages[0].page_number == 1
    assert len(result.document.pages[0].text_blocks) == 0
