import tempfile

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
