import pytest
from chunking.hasher import generate_ldu_hash

def test_hash_stability():
    """
    Rubric Requirement:
    Hash must remain valid even when page numbers shift across document versions.
    Doc_V1 (page 1) vs Doc_V2 (page 3) should produce the precise exact deterministic hash
    if content, chunk_type, and bounding box are equivalent.
    """
    
    content = "This is a stable content snippet."
    bbox = [0.1, 0.2, 0.8, 0.3]
    chunk_type = "text"
    
    hash_v1 = generate_ldu_hash(content, bbox, [1], chunk_type)
    hash_v2 = generate_ldu_hash(content, bbox, [3], chunk_type)
    
    assert hash_v1 == hash_v2, "Failure: Hash stability compromised by page number shift."
