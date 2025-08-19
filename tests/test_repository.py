import sqlite3

from sqlite_rag.models.chunk import Chunk
from sqlite_rag.models.document import Document
from sqlite_rag.repository import Repository


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

    def test_list_documents_empty(self, db_conn):
        conn, settings = db_conn

        repo = Repository(conn, settings)

        documents = repo.list_documents()

        assert len(documents) == 0

    def test_find_document_by_id_or_uri_by_id(self, db_conn):
        conn, settings = db_conn
        repo = Repository(conn, settings)

        # Add a document
        doc = Document(
            content="Test document content.",
            uri="test.txt",
            metadata={"author": "test"},
        )
        doc_id = repo.add_document(doc)

        # Find by ID
        found_doc = repo.find_document_by_id_or_uri(doc_id)

        assert found_doc is not None
        assert found_doc.id == doc_id
        assert found_doc.content == "Test document content."
        assert found_doc.uri == "test.txt"
        assert found_doc.metadata == {"author": "test"}

    def test_find_document_by_id_or_uri_by_uri(self, db_conn):
        conn, settings = db_conn
        repo = Repository(conn, settings)

        # Add a document
        doc = Document(
            content="Test document content.",
            uri="test.txt",
            metadata={"author": "test"},
        )
        repo.add_document(doc)

        # Find by URI
        found_doc = repo.find_document_by_id_or_uri("test.txt")

        assert found_doc is not None
        assert found_doc.content == "Test document content."
        assert found_doc.uri == "test.txt"
        assert found_doc.metadata == {"author": "test"}

    def test_find_document_by_id_or_uri_not_found(self, db_conn):
        conn, settings = db_conn
        repo = Repository(conn, settings)

        # Try to find non-existent document
        found_doc = repo.find_document_by_id_or_uri("nonexistent")

        assert found_doc is None

    def test_remove_document_success(self, db_conn):
        conn, settings = db_conn
        repo = Repository(conn, settings)

        # Add a document with chunks
        doc = Document(
            content="Test document content.",
            uri="test.txt",
            metadata={"author": "test"},
        )
        doc.chunks = [
            Chunk(content="Chunk 1", embedding=b"\x00" * 384),
            Chunk(content="Chunk 2", embedding=b"\x00" * 384),
        ]
        doc_id = repo.add_document(doc)

        # Verify document and chunks exist
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,))
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
        assert cursor.fetchone()[0] == 2

        # Remove document
        success = repo.remove_document(doc_id)

        assert success is True

        # Verify document and chunks are removed
        cursor.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,))
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
        assert cursor.fetchone()[0] == 0

    def test_remove_document_not_found(self, db_conn):
        conn, settings = db_conn
        repo = Repository(conn, settings)

        # Try to remove non-existent document
        success = repo.remove_document("nonexistent-id")

        assert success is False

    def test_document_exists_by_hash_exists(self, db_conn):
        conn, settings = db_conn
        repo = Repository(conn, settings)

        doc = Document(
            content="Test document content.",
            uri="test.txt",
            metadata={"author": "test"},
        )
        repo.add_document(doc)

        exists = repo.document_exists_by_hash(doc.hash())
        assert exists is True

    def test_document_exists_by_hash_not_exists(self, db_conn):
        conn, settings = db_conn
        repo = Repository(conn, settings)

        # Check for non-existent hash
        fake_doc = Document(content="Non-existent content")
        exists = repo.document_exists_by_hash(fake_doc.hash())

        assert exists is False
