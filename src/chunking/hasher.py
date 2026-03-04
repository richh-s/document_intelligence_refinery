"""Deterministic hashing for LDU provenance."""

import hashlib
from typing import List, Literal

def generate_ldu_hash(
    content: str, 
    bounding_box: List[float], 
    page_refs: List[int], 
    chunk_type: Literal["text", "table", "list", "figure", "header"]
) -> str:
    """
    Generates a deterministic spatial hash preventing collisions between 
    raw table chunks, table summaries, and descriptive text.
    
    Hash Strategy: hash(content + rounded_normalized_bbox + page_refs + chunk_type)
    Bounding boxes are rounded to 4 decimal places to avoid floating-point variance.
    """
    # Normalize bounding box by rounding to prevent minor FP discrepancies
    rounded_bbox = [round(v, 4) for v in bounding_box]
    
    # Construct the unique signature string
    signature_parts = [
        content.strip(),
        f"bbox:{rounded_bbox}",
        f"pages:{sorted(page_refs)}",
        f"type:{chunk_type}"
    ]
    
    signature = "|".join(signature_parts)
    
    # Generate SHA-256 hash
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()
