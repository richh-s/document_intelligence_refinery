"""Semantic Chunking and Logical Document Unit representations.

This package transforms raw extracted documents into RAG-safe Logical Document Units (LDUs).
"""

from models.ldu import LogicalDocumentUnit, LDUMetadata
from chunking.hasher import generate_ldu_hash
from chunking.validator import ChunkValidator, ChunkValidationError
from chunking.engine import ChunkingEngine

__all__ = [
    "LogicalDocumentUnit",
    "LDUMetadata",
    "generate_ldu_hash",
    "ChunkValidator",
    "ChunkValidationError",
    "ChunkingEngine",
]
