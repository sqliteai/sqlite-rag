import sqlite3
from pathlib import Path

from .database import Database
from .settings import Settings


class ConnectionRegistry:
    """Manage SQLite connections."""

    EMBEDDING = "embedding"
    TEXT_GEN = "text_generation"

    def __init__(
        self,
        db_path: str,
        base_connection: sqlite3.Connection,
        settings: Settings,
    ):
        self._settings = settings
        self._db_path = db_path

        self._connections: dict[str, sqlite3.Connection] = {}
        self._connections[self.EMBEDDING] = base_connection

    @classmethod
    def create(
        cls,
        db_path: str,
        settings: Settings,
        require_existing: bool = False,
    ) -> "ConnectionRegistry":
        if require_existing and not Path(db_path).exists():
            raise FileNotFoundError(f"Database file {db_path} does not exist.")

        conn = Database.new_connection(db_path)

        Database.load_exetension_ai(conn, settings)
        Database.load_exetension_vector(conn, settings)

        Database.initialize_schema(conn, settings)
        Database.initialize_vector_store(conn, settings)

        registry = cls(
            db_path,
            conn,
            settings,
        )

        return registry

    def get_connection(self, name: str) -> sqlite3.Connection:
        if name == self.EMBEDDING:
            return self.get_conn_embedding()
        if name == self.TEXT_GEN:
            return self.get_conn_text_gen()
        raise KeyError(f"Connection '{name}' is not supported.")

    def get_conn_embedding(self) -> sqlite3.Connection:
        if self.EMBEDDING not in self._connections:
            raise ValueError("Embedding connection not found.")

        return self._connections[self.EMBEDDING]

    def get_conn_text_gen(self) -> sqlite3.Connection:
        if self.TEXT_GEN in self._connections:
            return self._connections[self.TEXT_GEN]

        conn = Database.new_connection(self._db_path)
        Database.load_exetension_ai(conn, self._settings)

        self._connections[self.TEXT_GEN] = conn
        return conn

    def close(self, name: str) -> None:
        conn = self._connections.pop(name, None)
        if conn:
            conn.close()

    def close_all(self) -> None:
        for name in list(self._connections.keys()):
            self.close(name)
