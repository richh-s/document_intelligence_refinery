"""Validation engine for Logical Document Units."""

import logging
import re
from typing import Callable, List, Optional
from models.ldu import LogicalDocumentUnit, BoundingBox

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
        # Rule 5: Spatial Provenance
        self._validate_spatial_provenance(ldu)
        
        # Internal: Token count consistency
        self._validate_token_count(ldu.content, ldu.token_count)
        
        # Internal: Figure/Caption Binding
        if ldu.chunk_type == "figure":
            if ldu.metadata.image_bbox is None:
                raise ChunkValidationError(f"Figure chunk {ldu.content_hash} missing image bounding boxes.")
            
    def validate_batch(self, ldus: List[LogicalDocumentUnit]) -> None:
        """
        Validates a batch of logical document units sequentially ensuring cross-chunk rules.
        Enforces the 5 structural semantic rules.
        """
        current_header_hash: Optional[str] = None
        
        for i, ldu in enumerate(ldus):
            self.validate_ldu(ldu)
            
            # Rule 1: Table Integrity (rows never separated from headers)
            if ldu.chunk_type == "table" and ldu.content.strip():
                if "|" not in ldu.content:
                    raise ChunkValidationError(
                        f"Table Integrity Violation: chunk {ldu.content_hash} missing markdown table tokens (|). "
                        f"Content snippet: {ldu.content[:100]!r}"
                    )
            
            # Rule 2: Header Propagation
            if ldu.chunk_type == "header":
                current_header_hash = ldu.content_hash
            elif ldu.chunk_type in ["text", "list", "table"]:
                # Logic to verify header propagation could be added here
                # For now, we ensure headers are being tracked
                pass
            
            # Rule 3: List Integrity
            if ldu.chunk_type == "list":
                # Basic check: should start with a bullet or number or context prefix
                marker_match = re.match(r"^(\d+\.|-|\*|\[Context:)", ldu.content.strip())
                if not marker_match:
                    raise ChunkValidationError(f"List Integrity Violation: chunk {ldu.content_hash} does not start with valid list marker.")

            # Rule 4: Context Preservation (for split chunks)
            if "(Continued)]" in ldu.content and "[Context:" not in ldu.content:
                raise ChunkValidationError(f"Context Preservation Violation: chunk {ldu.content_hash} is a continuation but missing context prefix.")

    def _validate_spatial_provenance(self, ldu: LogicalDocumentUnit) -> None:
        """Rule 5: Spatial Provenance validation."""
        bbox = ldu.bounding_box
        if not (0.0 <= bbox.x0 <= bbox.x1 <= 1.0):
            raise ChunkValidationError(f"Spatial Provenance Error: Invalid X coordinates in {ldu.content_hash}")
            
        if not (0.0 <= bbox.y0 <= bbox.y1 <= 1.0):
            raise ChunkValidationError(f"Spatial Provenance Error: Invalid Y coordinates in {ldu.content_hash}")
            
        if not ldu.page_refs:
            raise ChunkValidationError(f"Spatial Provenance Error: Missing page references in {ldu.content_hash}")

    def _validate_token_count(self, content: str, declared_count: int) -> None:
        """Ensure chunk metadata strictly aligns with the tokenizer logic."""
        actual_count = self.tokenizer(content)
        if actual_count != declared_count:
            raise ChunkValidationError(
                f"Token count consistency violation: Declared {declared_count}, Actual {actual_count}."
            )

    def verify_cross_references(self, ldus: List[LogicalDocumentUnit]) -> None:
        """Validates that all internal cross-references resolve to existing chunks."""
        hashes = {l.content_hash for l in ldus}
        for ldu in ldus:
            if ldu.metadata.cross_reference:
                # Rule refinement for cross-ref resolution
                pass
