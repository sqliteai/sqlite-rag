"""
SQLite RAG - Hybrid search with SQLite AI and SQLite Vector

This package provides both a library interface and CLI for document 
indexing and semantic search using SQLite with AI and vector extensions.
"""

from .sqliterag import SQLiteRag
from .models.document import Document
from .models.chunk import Chunk
from .settings import Settings

__version__ = "0.1.0"
__all__ = [
    "SQLiteRag",
    "Document", 
    "Chunk",
    "Settings",
]