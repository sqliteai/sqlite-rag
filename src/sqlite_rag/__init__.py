"""
SQLite RAG - Hybrid search with SQLite AI and SQLite Vector

This package provides both a library interface and CLI for document
indexing and semantic search using SQLite with AI and vector extensions.
"""

from .models.chunk import Chunk
from .models.document import Document
from .settings import Settings
from .sqliterag import SQLiteRag

__version__ = "0.1.0"
__all__ = [
    "SQLiteRag",
    "Document",
    "Chunk",
    "Settings",
]
