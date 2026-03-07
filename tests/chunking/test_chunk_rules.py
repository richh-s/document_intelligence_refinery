import pytest
from agents.chunker import ChunkingEngine
from chunking.validator import ChunkValidator, ChunkValidationError
from models.ldu import LogicalDocumentUnit, BoundingBox, LDUMetadata

def mock_tokenizer(text):
    return len(text.split())

@pytest.fixture
def validator():
    return ChunkValidator(tokenizer_fn=mock_tokenizer)

def test_rule_1_table_header_integrity(validator):
    """Rule 1: Table header separation should be flagged."""
    # Table chunk missing '|' separators (enforced in validator)
    ldu = LogicalDocumentUnit(
        content="This is not a markdown table",
        chunk_type="table",
        page_refs=[1],
        bounding_box=BoundingBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9),
        token_count=6,
        content_hash="h1",
        metadata=LDUMetadata()
    )
    with pytest.raises(ChunkValidationError, match="Table Integrity Violation"):
        validator.validate_batch([ldu])

def test_rule_4_figure_caption_metadata(validator):
    """Rule 4: Figure chunks must have image and caption bboxes."""
    ldu = LogicalDocumentUnit(
        content="Caption",
        chunk_type="figure",
        page_refs=[1],
        bounding_box=BoundingBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9),
        token_count=1,
        content_hash="h2",
        metadata=LDUMetadata() # Missing image_bbox and caption_bbox
    )
    with pytest.raises(ChunkValidationError, match="missing image/caption bounding boxes"):
        validator.validate_ldu(ldu)

def test_rule_5_cross_reference_resolution(validator):
    """Rule 5: Dangling cross-references must be flagged."""
    ldu = LogicalDocumentUnit(
        content="See Table 1",
        chunk_type="text",
        page_refs=[1],
        bounding_box=BoundingBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9),
        token_count=3,
        content_hash="h3",
        metadata=LDUMetadata(dangling_reference="table_1")
    )
    with pytest.raises(ChunkValidationError, match="unresolved cross-reference"):
        validator.validate_ldu(ldu)
