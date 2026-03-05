"""Base interface for all extraction strategies."""

import time
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from pydantic import BaseModel, Field

from models.document_profile import DocumentProfile
from models.extracted_document import ExtractedDocument


class ExtractionResult(BaseModel):
    """Standardized output from any extraction strategy."""

    document: ExtractedDocument
    confidence: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0)
    time_ms: int = Field(ge=0)
    
    # Track granular metrics for the ledger
    model_name: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    pages_sent: int = 0
    
    # Routing transparency for rubric completeness
    escalation_occurred: bool = False
    strategies_attempted: list[str] = Field(default_factory=list)
    requires_human_review: bool = False


class PartialExtractionResult(ExtractionResult):
    """Returned when a Vision extractor hits a budget limit mid-document.
    
    Preserves all extracted pages paid for up to the budget cap.
    """
    pass


class BaseExtractor(ABC):
    """Abstract base class for all extraction strategies."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the extractor with dynamic rules/config."""
        self.config = config or {}

    @abstractmethod
    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult:
        """Execute the extraction strategy.
        
        Args:
            pdf_path: Absolute path to the source PDF.
            profile: Triage profile containing document metadata and hints.
            
        Returns:
            An ExtractionResult containing the normalized Document and confidence.
        """
        pass
    
    def _timer_start(self) -> float:
        return time.perf_counter()
        
    def _timer_end_ms(self, start_time: float) -> int:
        return int((time.perf_counter() - start_time) * 1000)
