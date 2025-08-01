from pathlib import Path
import sqlite3
from typing import Optional

from chunker import Chunker
from database import Database
from engine import Engine
from models.document import Document
from reader import FileReader
from repository import Repository
from settings import Settings


class SQLiteRag:
    def __init__(self, settings: Optional[Settings] = None):
        if settings is None:
            # TODO: load defaults or from the database
            settings = Settings(
                model_path_or_name="all-MiniLM-L6-v2", db_path="sqliterag.db"
            )

        self.settings = settings

        self._conn = sqlite3.connect(settings.db_path)

        self._repository = Repository(self._conn, settings)
        self._engine = Engine(self._conn, settings)
        self._chunker = Chunker(self._conn, settings)

        self.ready = False

    def _ensure_initialized(self):
        if not self.ready:
            Database.initialize(self._conn, self.settings)
            self._engine.load_model()

        self.ready = True

    def add(self, path: str):
        """Add the file content into the database"""
        self._ensure_initialized()

        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        # TODO: check the file extension
        content = FileReader.parse_file(Path(path))
        # TODO: include metadata extraction and mdx options (see our docsearch)
        document = Document(content=content, uri=path)
        chunks = self._chunker.chunk(document.content)
        document.chunks = chunks

        self._repository.add_document(document)
