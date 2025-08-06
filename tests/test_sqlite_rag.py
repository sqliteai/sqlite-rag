import tempfile
from pathlib import Path

from sqlite_rag import SQLiteRag


class TestSQLiteRag:
    def test_add_simple_text_file(self, db_settings):
        #  test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "This is a test document with some content. And this is another very long sentence."
            )
            temp_file_path = f.name

        rag = SQLiteRag(db_settings)

        rag.add(temp_file_path)

        conn = rag._conn
        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        assert doc_count == 1

        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0

    def test_add_directory(self, db_settings):
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "file1.txt"
            file2 = Path(temp_dir) / "file2.txt"

            file1.write_text("This is the first test document.")
            file2.write_text("This is the second test document.")

            db_settings.chunk_size = 100
            db_settings.chunk_overlap = 10

            rag = SQLiteRag(db_settings)

            rag.add(temp_dir)

            conn = rag._conn
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            assert doc_count == 2

            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            assert chunk_count > 0

    def test_add_text(self, db_settings):
        rag = SQLiteRag(db_settings)

        rag.add_text(
            "This is a test document content with some text to be indexed.",
            uri="test_doc.txt",
            metadata={"author": "test"},
        )

        conn = rag._conn
        cursor = conn.execute("SELECT content, uri, metadata FROM documents")
        doc = cursor.fetchone()
        assert doc
        assert doc[0] == "This is a test document content with some text to be indexed."
        assert doc[1] == "test_doc.txt"
        assert doc[2] == '{"author": "test"}'

        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0

    def test_add_text_without_options(self, db_settings):
        rag = SQLiteRag(db_settings)

        rag.add_text("This is a test document content without options.")

        conn = rag._conn
        cursor = conn.execute("SELECT content FROM documents")
        doc = cursor.fetchone()
        assert doc
        assert doc[0] == "This is a test document content without options."

        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0

    def test_list_documents(self, db_settings):
        rag = SQLiteRag(db_settings)

        rag.add_text("Document 1 content.")
        rag.add_text("Document 2 content.")

        documents = rag.list_documents()
        assert len(documents) == 2
        assert documents[0].content == "Document 1 content."
        assert documents[1].content == "Document 2 content."

    def test_find_document_by_id(self, db_settings):
        rag = SQLiteRag(db_settings)

        rag.add_text(
            "Test document content.", uri="test.txt", metadata={"author": "test"}
        )
        documents = rag.list_documents()
        doc_id = documents[0].id

        # Find by ID
        assert doc_id is not None
        found_doc = rag.find_document(doc_id)

        assert found_doc is not None
        assert found_doc.id == doc_id
        assert found_doc.content == "Test document content."
        assert found_doc.uri == "test.txt"
        assert found_doc.metadata == {"author": "test"}

    def test_find_document_by_uri(self, db_settings):
        rag = SQLiteRag(db_settings)

        rag.add_text(
            "Test document content.", uri="test.txt", metadata={"author": "test"}
        )

        # Find by URI
        found_doc = rag.find_document("test.txt")

        assert found_doc is not None
        assert found_doc.content == "Test document content."
        assert found_doc.uri == "test.txt"
        assert found_doc.metadata == {"author": "test"}

    def test_find_document_not_found(self, db_settings):
        rag = SQLiteRag(db_settings)

        found_doc = rag.find_document("nonexistent")

        assert found_doc is None

    def test_remove_document_by_id(self, db_settings):
        rag = SQLiteRag(db_settings)

        rag.add_text(
            "Test document content.", uri="test.txt", metadata={"author": "test"}
        )
        documents = rag.list_documents()
        doc_id = documents[0].id

        # Verify document exists
        assert len(documents) == 1

        # Remove by ID
        assert doc_id is not None
        success = rag.remove_document(doc_id)

        assert success is True

        # Verify document is removed
        documents = rag.list_documents()
        assert len(documents) == 0

    def test_remove_document_by_uri(self, db_settings):
        rag = SQLiteRag(db_settings)

        rag.add_text(
            "Test document content.", uri="test.txt", metadata={"author": "test"}
        )

        # Verify document exists
        documents = rag.list_documents()
        assert len(documents) == 1

        # Remove by URI
        success = rag.remove_document("test.txt")

        assert success is True

        # Verify document is removed
        documents = rag.list_documents()
        assert len(documents) == 0

    def test_remove_document_not_found(self, db_settings):
        rag = SQLiteRag(db_settings)

        success = rag.remove_document("nonexistent")

        assert success is False

    def test_remove_document_with_chunks(self, db_settings):
        rag = SQLiteRag(db_settings)

        # Add document that will create chunks
        rag.add_text(
            "This is a longer document that should create multiple chunks when processed by the chunker.",
            uri="test.txt",
        )

        # Verify document and chunks exist
        documents = rag.list_documents()
        assert len(documents) == 1
        doc_id = documents[0].id

        cursor = rag._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0

        # Remove document
        assert doc_id is not None
        success = rag.remove_document(doc_id)

        assert success is True

        # Verify document and chunks are removed
        documents = rag.list_documents()
        assert len(documents) == 0

        cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
        chunk_count = cursor.fetchone()[0]
        assert chunk_count == 0
