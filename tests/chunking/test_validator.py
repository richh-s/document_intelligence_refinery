import pytest
from models.ldu import LogicalDocumentUnit, LDUMetadata
from chunking.validator import ChunkValidator, ChunkValidationError
from chunking.hasher import generate_ldu_hash

def dummy_tokenizer(text: str) -> int:
    # simple split by space for testing
    return len(text.split())

def test_chunk_validator_bbox_checks():
    validator = ChunkValidator(tokenizer_fn=dummy_tokenizer)
    
    # Valid BBox
    valid_ldu = LogicalDocumentUnit(
        content="Hello world",
        chunk_type="text",
        page_refs=[1],
        bounding_box=[0.1, 0.1, 0.9, 0.9],
        token_count=2,
        content_hash="abc",
        metadata=LDUMetadata()
    )
    validator.validate_ldu(valid_ldu) # Should not raise
    
    # Invalid BBox (out of bounds)
    invalid_ldu = valid_ldu.model_copy(update={"bounding_box": [-0.1, 0.1, 0.9, 0.9]})
    with pytest.raises(ChunkValidationError, match="Must satisfy 0.0 <= x0 < x1 <= 1.0"):
        validator.validate_ldu(invalid_ldu)
        
    # Invalid BBox (x0 >= x1)
    invalid_ldu_2 = valid_ldu.model_copy(update={"bounding_box": [0.9, 0.1, 0.1, 0.9]})
    with pytest.raises(ChunkValidationError, match="Must satisfy 0.0 <= x0 < x1 <= 1.0"):
        validator.validate_ldu(invalid_ldu_2)

def test_chunk_validator_token_count():
    validator = ChunkValidator(tokenizer_fn=dummy_tokenizer)
    
    ldu = LogicalDocumentUnit(
        content="Test token count validation process.",
        chunk_type="text",
        page_refs=[1],
        bounding_box=[0.1, 0.1, 0.9, 0.9],
        token_count=100, # Deliberately wrong (actual is 5)
        content_hash="abc",
        metadata=LDUMetadata()
    )
    
    with pytest.raises(ChunkValidationError, match="Token count consistency violation"):
        validator.validate_ldu(ldu)

def test_validator_table_span_stress_test():
    validator = ChunkValidator(tokenizer_fn=dummy_tokenizer)
    
    ldu = LogicalDocumentUnit(
        content="Row 1\nRow 2",
        chunk_type="table",
        page_refs=[1, 2],
        bounding_box=[0.1, 0.1, 0.9, 0.9],
        token_count=4,
        content_hash="abc",
        metadata=LDUMetadata()
    )
    
    # Simulate a table spanning multiple pages (page_refs has length > 1) 
    # but missing injected headers (has_headers=False).
    with pytest.raises(ChunkValidationError, match="missing injected headers"):
        validator.enforce_table_header_injection(ldu, is_spanning_table=True, has_headers=False)

def test_spatial_hash_shift():
    content = "Some extracted text"
    page_refs = [1]
    chunk_type = "text"
    
    bbox1 = [0.1000, 0.2000, 0.5000, 0.6000]
    hash1 = generate_ldu_hash(content, bbox1, page_refs, chunk_type)
    
    # Small shift in spatial coordinates (e.g., page element moved slightly)
    bbox2 = [0.1001, 0.2000, 0.5000, 0.6000]
    hash2 = generate_ldu_hash(content, bbox2, page_refs, chunk_type)
    
    # Hash should be completely different
    assert hash1 != hash2, "Spatial hash must change when bbox changes"
    
    # Shift small enough to round to same 4-decimal points? No, 0.1001 vs 0.1000 is different at 4th decimal.
    # Let's test floating point variance safety (e.g., 0.10000001 vs 0.10000002 should hash to the same value)
    bbox_fp_1 = [0.100041, 0.2, 0.5, 0.6]
    bbox_fp_2 = [0.100042, 0.2, 0.5, 0.6]
    
    hash_fp_1 = generate_ldu_hash(content, bbox_fp_1, page_refs, chunk_type)
    hash_fp_2 = generate_ldu_hash(content, bbox_fp_2, page_refs, chunk_type)
    
    assert hash_fp_1 == hash_fp_2, "Spatial hash should ignore tiny floating point variances."
