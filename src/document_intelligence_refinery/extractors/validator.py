"""Validation layer to enforce quality and structure post-extraction."""

from typing import Any
from document_intelligence_refinery.schema import ExtractedDocument
from document_intelligence_refinery.models import DocumentProfile


class ExtractionValidator:
    """Performs post-extraction health checks."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        # Ensure rules exist, use defaults if missing
        self.page_continuity_penalty = float(self.config.get("PAGE_CONTINUITY_PENALTY", 0.4))
        
    def validate(self, document: ExtractedDocument, profile: DocumentProfile) -> float:
        """Run all health checks on the extracted document.
        
        Returns:
            validator_confidence (float): A score between 0.0 and 1.0. 
            Used by the Router as: final_confidence = min(extractor_conf, validator_conf).
        """
        confidence = 1.0
        
        # 1. Ghost Page Continuity Check (Strict)
        # If the extractor dropped pages or crashed mid-run, force immediate escalation.
        actual_pages = len(document.pages)
        expected_pages = profile.page_count
        
        if actual_pages != expected_pages:
            # We strictly enforce the penalty. E.g., 0.95 extractor conf * 0.4 penalty = 0.38
            # which will trigger the < 0.85 MIN_EXTRACTION_CONFIDENCE router gate.
            confidence *= self.page_continuity_penalty
            
            # If the difference is severe (missing more than half), kill confidence entirely
            if actual_pages < (expected_pages / 2):
                confidence = 0.0
                return confidence
                
        # 2. Text Content Check
        # If a page claims to have blocks but none have text, it's a parse failure.
        empty_pages = 0
        for page in document.pages:
            has_blocks = len(page.text_blocks) > 0
            has_content = any(len(b.text.strip()) > 0 for b in page.text_blocks)
            
            if has_blocks and not has_content:
                empty_pages += 1
                
        if empty_pages > 0 and actual_pages > 0:
            # Penalize linearly based on percentage of blank text blocks
            penalty = 1.0 - (empty_pages / actual_pages)
            confidence *= penalty

        # Minimum bounds enforcement
        return max(0.0, min(1.0, confidence))
