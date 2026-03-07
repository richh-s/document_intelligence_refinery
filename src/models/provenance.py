"""Provenance Layer Schema."""

from typing import List, Optional
from pydantic import BaseModel, Field

class ProvenanceChain(BaseModel):
    """
    Guarantees strict mathematical linkage between synthesized answers 
    and their source extraction chunks to prevent hallucination.
    """
    document_name: str = Field(..., description="Source document name")
    page_number: int = Field(..., description="Target page number (1-indexed)")
    bbox: List[float] = Field(..., description="Normalized spatial coordinates [x0, y0, x1, y1]")
    content_hash: str = Field(..., description="Exact deterministic LDU ID")
    text_snippet: str = Field(..., max_length=500, description="Extremely dense excerpt (30-50 words max)")
    
    # Contextual Qualifiers
    section_title: Optional[str] = Field(None, description="The structural chapter this chunk belongs to")
    
    @classmethod
    def from_ldu(cls, ldu: 'LogicalDocumentUnit', doc_name: str, section_title: Optional[str] = None) -> 'ProvenanceChain':
        """Constructs a strict Provenance object dynamically from an LDU."""
        bbox_list = [ldu.bounding_box.x0, ldu.bounding_box.y0, ldu.bounding_box.x1, ldu.bounding_box.y1]
        
        # Enforce lean snippets (approx 30 words max based on words)
        words = ldu.content.split()
        snippet = " ".join(words[:30])
        if len(words) > 30:
            snippet += "..."
            
        return cls(
            document_name=doc_name,
            page_number=ldu.page_refs[0] if ldu.page_refs else 1,
            bbox=bbox_list,
            content_hash=ldu.content_hash,
            text_snippet=snippet,
            section_title=section_title
        )
