"""Validation engine for Logical Document Units."""

import logging
from typing import Callable, List
from models.ldu import LogicalDocumentUnit

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

    def _validate_bounding_box(self, bbox: List[float]) -> None:
        if len(bbox) != 4:
            raise ChunkValidationError(f"Bounding box must have 4 float values, got {len(bbox)}: {bbox}")
            
        x0, y0, x1, y1 = bbox
        
        if not (0.0 <= x0 < x1 <= 1.0):
            raise ChunkValidationError(f"Invalid X coordinates. Must satisfy 0.0 <= x0 < x1 <= 1.0. Got x0={x0}, x1={x1}")
            
        if not (0.0 <= y0 < y1 <= 1.0):
            raise ChunkValidationError(f"Invalid Y coordinates. Must satisfy 0.0 <= y0 < y1 <= 1.0. Got y0={y0}, y1={y1}")

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
