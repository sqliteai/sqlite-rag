import sqlite3
import tempfile

from sqlite_rag.database import Database
from sqlite_rag.settings import Settings


class TestDatabase:
    def test_db_initialization(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
            settings = Settings(
                model_path_or_name="all-MiniLM-L6-v2", db_path=tmp_db.name
            )

        conn = sqlite3.connect(settings.db_path)
        Database.initialize(conn, settings)

        # Check if the tables exist
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        assert cursor.fetchone() is not None, "Documents table was not created."

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        )
        assert cursor.fetchone() is not None, "Chunks table was not created."
