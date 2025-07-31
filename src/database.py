import sqlite3
from pathlib import Path

from settings import Settings


class Database:
    """Database initialization and schema management for SQLiteRag."""

    @staticmethod
    def initialize(conn: sqlite3.Connection, settings: Settings) -> sqlite3.Connection:
        """Initialize the database with extensions and schema"""
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(Path(__file__).parent.parent / "extensions" / "ai"))
            conn.load_extension(
                str(Path(__file__).parent.parent / "extensions" / "vector")
            )
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "Failed to load extensions: "
                + str(e)
                + """\n
                Download from:
                    sqlite-ai: https://github.com/sqliteai/sqlite-ai/releases
                    sqlite-vector: https://github.com/sqliteai/sqlite-vector/releases
                """
            ) from e
        conn.enable_load_extension(False)

        try:
            # Check if extensions are available
            conn.execute("SELECT vector_version()")
            conn.execute("SELECT ai_version()")
        except sqlite3.OperationalError:
            raise RuntimeError("Extensions are not loaded correctly.")

        Database._create_schema(conn, settings)

        return conn

    @staticmethod
    def _create_schema(conn: sqlite3.Connection, settings: Settings):
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                hash TEXT NOT NULL UNIQUE,
                uri TEXT,
                content TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )

        # TODO: this table is not ready for sqlite-sync, it uses the id AUTOINCREMENT
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                content TEXT,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """
        )

        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, content='chunks', content_rowid='id');
        """
        )

        cursor.execute(
            f"""
            SELECT vector_init('chunks', 'embedding', 'type={settings.vector_type},dimension={settings.embedding_dim}');
        """
        )

        conn.commit()
