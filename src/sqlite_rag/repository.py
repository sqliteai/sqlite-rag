import json
import sqlite3
from uuid import uuid4

from .models.document import Document
from .settings import Settings


class Repository:
    def __init__(self, conn: sqlite3.Connection, settings: Settings):
        self._conn = conn
        self.settings = settings

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

    def list_documents(self) -> list[Document]:
        """List all documents in the database"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id, content, uri, metadata FROM documents")
        rows = cursor.fetchall()

        documents = []
        for row in rows:
            doc_id, content, uri, metadata = row
            documents.append(
                Document(
                    id=doc_id,
                    content=content,
                    uri=uri,
                    metadata=json.loads(metadata),
                )
            )

        return documents
