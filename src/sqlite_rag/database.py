import importlib
import importlib.resources
import sqlite3

from .settings import Settings


class Database:
    """Database initialization and schema management for SQLiteRag."""

    @staticmethod
    def new_connection(db_path: str = "./sqliterag.sqlite") -> sqlite3.Connection:
        """Create a new SQLite connection to the specified database path."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def initialize(conn: sqlite3.Connection, settings: Settings) -> sqlite3.Connection:
        """Initialize the database with extensions and schema"""
        conn.enable_load_extension(True)
        try:
            conn.load_extension(
                str(
                    importlib.resources.files(
                        "sqliteai.binaries." + ("gpu" if settings.use_gpu else "cpu")
                    )
                    / "ai"
                )
            )
            conn.load_extension(
                str(importlib.resources.files("sqlite_vector.binaries") / "vector")
            )
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "Failed to load extensions: "
                + str(e)
                + """\n
                Install via pip:
                    pip install sqlite-ai sqliteai-vector

                See more:
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
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                content TEXT,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
        """
        )

        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS sentences (
                id TEXT PRIMARY KEY,
                chunk_id INTEGER,
                content TEXT,
                embedding BLOB,
                start_offset INTEGER,
                end_offset INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_sentences_chunk_id ON sentences (chunk_id);
            """
        )

        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, content='chunks', content_rowid='id');
        """
        )

        cursor.execute(
            """
            SELECT vector_init('chunks', 'embedding', ?);
        """,
            (settings.get_vector_init_options(),),
        )
        # TODO: same configuration as chunks (or different options?)
        cursor.execute(
            """
            SELECT vector_init('sentences', 'embedding', ?);
        """,
            (settings.get_vector_init_options(),),
        )

        conn.commit()
