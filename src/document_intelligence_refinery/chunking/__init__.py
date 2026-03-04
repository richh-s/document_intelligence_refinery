"""Semantic Chunking and Logical Document Unit representations.

This package transforms raw extracted documents into RAG-safe Logical Document Units (LDUs).
"""

from document_intelligence_refinery.chunking.models import LogicalDocumentUnit, LDUMetadata
from document_intelligence_refinery.chunking.hasher import generate_ldu_hash
from document_intelligence_refinery.chunking.validator import ChunkValidator, ChunkValidationError
from document_intelligence_refinery.chunking.engine import ChunkingEngine

__all__ = [
    "LogicalDocumentUnit",
    "LDUMetadata",
    "generate_ldu_hash",
    "ChunkValidator",
    "ChunkValidationError",
    "ChunkingEngine",
]
