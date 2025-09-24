import json
import sqlite3
from uuid import uuid4

from .models.document import Document
from .settings import Settings


class Repository:
    def __init__(self, conn: sqlite3.Connection, settings: Settings):
        self._conn = conn
        self._settings = settings

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
            cursor.execute(
                "INSERT INTO chunks (document_id, content, embedding) VALUES (?, ?, ?)",
                (document_id, chunk.content, chunk.embedding),
            )
            cursor.execute(
                "INSERT INTO chunks_fts (rowid, content) VALUES (last_insert_rowid(), ?)",
                (chunk.content,),
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

    def find_document_by_id_or_uri(self, identifier: str) -> Document | None:
        """Find document by ID or URI"""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT id, content, uri, metadata, created_at FROM documents WHERE id = ? OR uri = ?",
            (identifier, identifier),
        )
        row = cursor.fetchone()

        if row:
            doc_id, content, uri, metadata, created_at = row
            return Document(
                id=doc_id,
                content=content,
                uri=uri,
                metadata=json.loads(metadata),
                created_at=created_at,
            )
        return None

    def document_exists_by_hash(self, hash: str) -> bool:
        """Check if a document with the given hash exists"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM documents WHERE hash = ?", (hash,))
        return cursor.fetchone() is not None

    def remove_document(self, document_id: str) -> bool:
        """Remove document and its chunks by document ID"""
        cursor = self._conn.cursor()

        # Check if document exists
        cursor.execute(
            "SELECT COUNT(*) AS total FROM documents WHERE id = ?", (document_id,)
        )
        if cursor.fetchone()["total"] == 0:
            return False

        # Remove chunks first
        cursor.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunks WHERE document_id = ?)",
            (document_id,),
        )
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

        # Remove document
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

        self._conn.commit()
        return True
