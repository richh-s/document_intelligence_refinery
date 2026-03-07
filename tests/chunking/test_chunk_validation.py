import pytest
import re
from typing import List
from chunking.validator import ChunkValidator, ChunkValidationError
from models.ldu import LogicalDocumentUnit, LDUMetadata, BoundingBox

def dummy_tokenizer(text: str) -> int:
    return len(text.split())

@pytest.fixture
def validator():
    return ChunkValidator(tokenizer_fn=dummy_tokenizer)

def create_ldu(content: str, chunk_type: str, page_refs: List[int] = [1], bbox: List[float] = [0.1, 0.1, 0.9, 0.9]) -> LogicalDocumentUnit:
    return LogicalDocumentUnit(
        content=content,
        chunk_type=chunk_type,
        page_refs=page_refs,
        bounding_box={"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
        token_count=dummy_tokenizer(content),
        content_hash=f"hash_{hash(content)}",
        metadata=LDUMetadata()
    )

def test_rule_1_table_integrity(validator):
    # Missing markdown structure
    bad_table = create_ldu("This is not a table", "table")
    with pytest.raises(ChunkValidationError, match="Table Integrity Violation"):
        validator.validate_batch([bad_table])
    
    # Valid markdown table
    good_table = create_ldu("| Header |\n| --- |\n| Row |", "table")
    validator.validate_batch([good_table]) # Should pass

def test_rule_3_list_integrity(validator):
    # Bad list marker
    bad_list = create_ldu("Plain text starting list", "list")
    with pytest.raises(ChunkValidationError, match="List Integrity Violation"):
        validator.validate_batch([bad_list])
    
    # Valid list marker
    good_list = create_ldu("* Item 1", "list")
    validator.validate_batch([good_list])

def test_rule_4_context_preservation(validator):
    # Continuation without context prefix but starts with a valid marker
    bad_cont = create_ldu("* This is a (Continued)] fragment", "list")
    # Wrap in validate_batch to trigger the check
    with pytest.raises(ChunkValidationError, match="Context Preservation Violation"):
        validator.validate_batch([bad_cont])
    
    # Valid context prefix
    good_cont = create_ldu("[Context: Item 1 (Continued)]\n* Sub-item", "list")
    validator.validate_batch([good_cont])

def test_rule_5_spatial_provenance(validator):
    # Invalid coordinates (x0 > x1)
    # Using values that pass Pydantic's [0,1] range but fail the x0 <= x1 rule
    bad_bbox = create_ldu("Valid text", "text", bbox=[0.9, 0.1, 0.1, 0.9])
    with pytest.raises(ChunkValidationError, match="Spatial Provenance Error"):
        validator.validate_ldu(bad_bbox)
    
    # Missing page refs
    no_pages = create_ldu("Valid text", "text", page_refs=[])
    with pytest.raises(ChunkValidationError, match="Spatial Provenance Error"):
        validator.validate_ldu(no_pages)

def test_token_consistency(validator):
    ldu = create_ldu("Three words here", "text")
    ldu.token_count = 100 # Manually corrupt
    with pytest.raises(ChunkValidationError, match="Token count consistency violation"):
        validator.validate_ldu(ldu)
