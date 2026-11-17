"""Chunking package for structured document processing."""

from backend.chunking.chunker import (
    DocumentChunk,
    chunk_document,
)

__all__ = [
    "DocumentChunk",
    "chunk_document",
]
