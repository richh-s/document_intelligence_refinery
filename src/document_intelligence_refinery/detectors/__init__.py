"""Triage detectors package.

Re-exports all detector classes for convenient imports.
"""

from document_intelligence_refinery.detectors.domain import (
    DomainHintClassifier,
    DomainClassificationStrategy,
    KeywordDomainStrategy,
)
from document_intelligence_refinery.detectors.layout import LayoutComplexityDetector
from document_intelligence_refinery.detectors.origin import OriginTypeDetector

__all__ = [
    "DomainClassificationStrategy",
    "DomainHintClassifier",
    "KeywordDomainStrategy",
    "LayoutComplexityDetector",
    "OriginTypeDetector",
]
