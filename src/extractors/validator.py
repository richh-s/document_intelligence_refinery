"""Validation layer to enforce quality and structure post-extraction."""

from typing import Any
from models.extracted_document import ExtractedDocument
from models.document_profile import DocumentProfile


class ExtractionValidator:
    """Performs post-extraction health checks."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        # Ensure rules exist, use defaults if missing
        self.page_continuity_penalty = float(self.config.get("PAGE_CONTINUITY_PENALTY", 0.4))
        
    def validate(self, document: ExtractedDocument, profile: DocumentProfile) -> tuple[float, str]:
        """Run all health checks on the extracted document.
        
        Returns:
            validator_score (float): 0.0 to 1.0
            status_flag (str): "HEALTHY" | "LOW_CONFIDENCE"
        """
        actual_pages = len(document.pages)
        expected_pages = profile.page_count
        
        # 1. Completeness Ratio
        completeness = actual_pages / max(1, expected_pages)
        
        # 2. Text Content Check
        empty_pages = 0
        for page in document.pages:
            has_blocks = len(page.text_blocks) > 0
            has_content = any(len(b.text.strip()) > 0 for b in page.text_blocks)
            if has_blocks and not has_content:
                empty_pages += 1
                
        content_health = 1.0 - (empty_pages / max(1, actual_pages))
        
        # Final Validator Score (proportional)
        score = completeness * content_health
        
        status = "HEALTHY"
        if completeness < 0.5 or content_health < 0.5:
            status = "LOW_CONFIDENCE"
            
        return max(0.0, min(1.0, score)), status
