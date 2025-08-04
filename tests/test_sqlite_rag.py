import tempfile
from pathlib import Path

from settings import Settings
from src.sqliterag import SQLiteRag


class TestSQLiteRag:
    def test_add_simple_text_file(self):
        #  test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "This is a test document with some content. And this is another very long sentence."
            )
            temp_file_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_file_path = f.name

        settings = Settings(
            model_path_or_name="./capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
            db_path=db_file_path,
        )
        settings.chunk_size = 2
        settings.chunk_overlap = 0

        rag = SQLiteRag(settings)

        rag.add(temp_file_path)

        conn = rag._conn
        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        assert doc_count == 1

        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count == 15

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
            assert chunk_count == 6

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