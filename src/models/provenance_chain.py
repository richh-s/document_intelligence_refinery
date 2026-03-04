"""Provenance chain tracking schema."""

from pydantic import BaseModel
from models.ldu import BoundingBox

class ProvenanceChain(BaseModel):
    """Provenance citation chain linking LDUs back to the source."""
    document_name: str
    page_number: int
    bbox: BoundingBox
    content_hash: str
