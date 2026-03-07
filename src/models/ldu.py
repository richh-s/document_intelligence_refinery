"""Logical Document Unit (LDU) Data Contract."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class LDUMetadata(BaseModel):
    """Structured optional metadata for tracking relations and references."""
    related_table_id: Optional[str] = None
    cross_reference: Optional[str] = None
    cross_reference_type: Optional[str] = None
    dangling_reference: Optional[str] = None
    chunk_relationships: List[Dict[str, str]] = Field(default_factory=list, description="Explicit chunk relationships")
    
    # Figure Metadata
    image_bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    caption_bbox: Optional[List[float]] = None

class BoundingBox(BaseModel):
    """Structured sub-model for spatial normalization."""
    x0: float = Field(..., ge=0.0, le=1.0)
    y0: float = Field(..., ge=0.0, le=1.0)
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)

class LogicalDocumentUnit(BaseModel):
    """
    The normalized Semantic Chunk representation optimized for RAG.
    
    A semantically coherent, self-contained unit that preserves structural context.
    """
    content: str
    chunk_type: Literal["text", "table", "list", "figure", "header"]
    page_refs: List[int]
    bounding_box: BoundingBox
    parent_section: Optional[str] = None
    token_count: int
    content_hash: str
    metadata: LDUMetadata
