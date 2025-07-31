import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from database import Database
from models.document import Document
from settings import Settings


class Repository:
    def __init__(self, conn: sqlite3.Connection, settings: Settings):
        self._conn = conn
        self.settings = settings

    def load_model(self, path: str):
        """Load the LLMA model"""
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self._conn.execute(
            f"SELECT llm_model_load('{model_path}', '{self.settings.model_config}');"
        )

    def add_document(self, document: Document) -> str:
        """Add a text content to the database"""
        cursor = self._conn.cursor()

        document_id = str(uuid4())
        cursor.execute(
            "INSERT INTO documents (id, hash, content, uri, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (
                document_id,
                document.hash(),
                document.content,
                document.uri,
                json.dumps(document.metadata),
            ),
        )

        for chunk in document.chunks:
            # TODO: use the right vector_convert function based on the vector type
            cursor.execute(
                "INSERT INTO chunks (document_id, content, embedding) VALUES (?, ?, vector_convert_f32(?))",
                (document_id, chunk.content, chunk.embedding),
            )

        self._conn.commit()
        
        return document_id
