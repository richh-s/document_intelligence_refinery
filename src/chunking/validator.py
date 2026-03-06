"""Validation engine for Logical Document Units."""

import logging
from typing import Callable, List
from models.ldu import LogicalDocumentUnit, BoundingBox
from chunking.hasher import generate_ldu_hash

logger = logging.getLogger(__name__)

class ChunkValidationError(Exception):
    """Raised when an LDU violates the core chunking rules."""
    pass

class ChunkValidator:
    """
    Enforces the semantic chunking constitution before LDUs are emitted 
    into the Vector Store.
    """
    
    def __init__(self, tokenizer_fn: Callable[[str], int]):
        """
        Accepts a tokenizer function (str -> int) for measuring token counts.
        """
        self.tokenizer = tokenizer_fn
        
    def validate_ldu(self, ldu: LogicalDocumentUnit) -> None:
        """
        Runs constraint checks on a single LDU.
        Raises ChunkValidationError if any strict rule is violated.
        """
        self._validate_bounding_box(ldu.bounding_box)
        self._validate_token_count(ldu.content, ldu.token_count)
        
        # Cross-reference metadata check (Dangling vs Resolved)
        if ldu.metadata.cross_reference and not ldu.metadata.cross_reference_type:
            logger.warning(f"LDU {ldu.content_hash} has cross_reference without type.")
            
    def validate_batch(self, ldus: List[LogicalDocumentUnit]) -> None:
        """Validates a batch of logical document units sequentially."""
        for ldu in ldus:
            self.validate_ldu(ldu)

    def _validate_bounding_box(self, bbox: BoundingBox) -> None:
        if not (0.0 <= bbox.x0 <= bbox.x1 <= 1.0):
            raise ChunkValidationError(f"Invalid X coordinates. Must satisfy 0.0 <= x0 < x1 <= 1.0. Got x0={bbox.x0}, x1={bbox.x1}")
            
        if not (0.0 <= bbox.y0 <= bbox.y1 <= 1.0):
            raise ChunkValidationError(f"Invalid Y coordinates. Must satisfy 0.0 <= y0 < y1 <= 1.0. Got y0={bbox.y0}, y1={bbox.y1}")

    def _validate_token_count(self, content: str, declared_count: int) -> None:
        """Ensure chunk metadata strictly aligns with the tokenizer logic."""
        actual_count = self.tokenizer(content)
        if actual_count != declared_count:
            raise ChunkValidationError(
                f"Token count consistency violation: Declared {declared_count}, Actual {actual_count}. "
                f"Content preview: {content[:50]}"
            )

    def enforce_table_header_injection(self, ldu: LogicalDocumentUnit, is_spanning_table: bool, has_headers: bool) -> None:
        """
        Post-processing rule: Enforces that chunks spanning multiple pages for a single
        table must contain header rows to remain conceptually coherent.
        """
        if ldu.chunk_type == "table" and is_spanning_table and not has_headers:
            raise ChunkValidationError(
                f"Table Integrity Violation: Spanning table chunk {ldu.content_hash} is missing injected headers."
            )

    def verify_post_transformation_hash(self, original_hash: str, transformed_ldu: LogicalDocumentUnit) -> None:
        """
        Re-asserts the content_hash post-transformation to guarantee text cleaning 
        (e.g., removing extra whitespace or special chars) does not invisibly destroy 
        core semantic data by dropping a 'not' or a decimal point.
        """
        bbox_list = [transformed_ldu.bounding_box.x0, transformed_ldu.bounding_box.y0, transformed_ldu.bounding_box.x1, transformed_ldu.bounding_box.y1]
        new_hash = generate_ldu_hash(
            content=transformed_ldu.content,
            bounding_box=bbox_list,
            page_refs=transformed_ldu.page_refs,
            chunk_type=transformed_ldu.chunk_type
        )
        if new_hash != original_hash:
            raise ChunkValidationError(
                f"Post-Transformation Hash Verification failed! Original hash {original_hash} "
                f"does not match new hash {new_hash} after text cleaning. Semantic data may be compromised."
            )
