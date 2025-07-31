import sqlite3

from models.chunk import Chunk


class Chunker:
    def __init__(self, conn: sqlite3.Connection, settings):
        self._conn = conn
        self.settings = settings

    def chunk(self, text: str) -> list[Chunk]:
        # Implement chunking logic here
        return []
