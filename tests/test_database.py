import sqlite3

from sqlite_rag.database import Database
from sqlite_rag.settings import Settings


class TestDatabase:
    def test_db_initialization(self):
        conn = sqlite3.connect(":memory")
        Database.initialize(conn, Settings())

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

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        assert cursor.fetchone() is not None, "Chunks table for FTS was not created."

        conn.execute("SELECT vector_version()")
        conn.execute("SELECT ai_version()")
