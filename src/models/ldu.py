"""Logical Document Unit (LDU) Data Contract."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class LDUMetadata(BaseModel):
    """Structured optional metadata for tracking relations and references."""
    related_table_id: Optional[str] = None
    cross_reference: Optional[str] = None
    cross_reference_type: Optional[str] = None
    dangling_reference: Optional[str] = None
    relations: List[Dict[str, str]] = Field(default_factory=list)

class LogicalDocumentUnit(BaseModel):
    """
    The normalized Semantic Chunk representation optimized for RAG.
    
    A semantically coherent, self-contained unit that preserves structural context.
    """
    content: str
    chunk_type: Literal["text", "table", "list", "figure", "header"]
    page_refs: List[int]
    bounding_box: List[float] = Field(..., description="[x0, y0, x1, y1] normalized 0-1")
    parent_section_id: Optional[str] = None
    token_count: int
    content_hash: str
    metadata: LDUMetadata
