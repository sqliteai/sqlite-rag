import sqlite3
import tempfile

from h11 import Data

from database import Database
from models.chunk import Chunk
from models.document import Document
from repository import Repository
from settings import Settings


class TestRepository:
    def test_add_document_without_chunks(self, db_conn):
        conn, settings = db_conn

        repo = Repository(conn, settings)

        doc_id = repo.add_document(
            Document(
                content="This is a test document content.",
                uri="test_doc.txt",
                metadata={"author": "test"},
            )
        )

        # Verify the document was added
        conn = sqlite3.connect(settings.db_path)
        cursor = conn.execute(
            "SELECT content, uri, metadata FROM documents WHERE id=?", (doc_id,)
        )
        row = cursor.fetchone()

        assert row is not None, "Document was not added to the database."
        assert row[0] == "This is a test document content."
        assert row[1] == "test_doc.txt"
        assert row[2] == '{"author": "test"}'

    def test_add_document_with_chunks(self, db_conn):
        conn, settings = db_conn

        repo = Repository(conn, settings)

        doc = Document(
            content="This is a test document with chunks.",
            uri="test_doc_with_chunks.txt",
            metadata={"author": "test"},
        )

        doc.chunks = [
            Chunk(content="Chunk 1 content", embedding=b"\x00" * 384),
            Chunk(content="Chunk 2 content", embedding=b"\x00" * 384),
        ]

        doc_id = repo.add_document(doc)

        # Verify the document and chunks were added
        conn = sqlite3.connect(settings.db_path)
        cursor = conn.execute(
            "SELECT content, uri, metadata FROM documents WHERE id=?", (doc_id,)
        )
        row = cursor.fetchone()

        assert row is not None, "Document was not added to the database."
        assert row[0] == "This is a test document with chunks."
        assert row[1] == "test_doc_with_chunks.txt"
        assert row[2] == '{"author": "test"}'

        cursor.execute(
            "SELECT content, embedding FROM chunks WHERE document_id=?", (doc_id,)
        )
        chunk_rows = cursor.fetchall()

        assert len(chunk_rows) == 2, "Chunks were not added to the database."
        assert chunk_rows[0][0] == "Chunk 1 content"
        assert chunk_rows[0][1] == b"\x00" * 384
        assert chunk_rows[1][0] == "Chunk 2 content"
        assert chunk_rows[1][1] == b"\x00" * 384

    def test_list_documents(self, db_conn):
        conn, settings = db_conn

        repo = Repository(conn, settings)

        doc1 = Document(
            content="Document 1 content.", uri="doc1.txt", metadata={"author": "test1"}
        )
        doc2 = Document(
            content="Document 2 content.", uri="doc2.txt", metadata={"author": "test2"}
        )

        repo.add_document(doc1)
        repo.add_document(doc2)

        documents = repo.list_documents()

        assert len(documents) == 2
        assert documents[0].id is not None
        assert documents[0].uri == "doc1.txt"
        assert documents[0].content == "Document 1 content."
        assert documents[0].metadata == {"author": "test1"}

        assert documents[1].id is not None
        assert documents[1].content == "Document 2 content."
        assert documents[1].uri == "doc2.txt"
        assert documents[1].metadata == {"author": "test2"}
