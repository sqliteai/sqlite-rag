import glob
from pathlib import Path
import sqlite3
from typing import Optional

from .chunker import Chunker
from .database import Database
from .engine import Engine
from .models.document import Document
from .reader import FileReader
from .repository import Repository
from .settings import Settings


class SQLiteRag:
    def __init__(self, settings: Optional[Settings] = None):
        if settings is None:
            # TODO: load defaults or from the database
            settings = Settings(
                model_path_or_name="./all-MiniLM-L6-v2.e4ce9877.q8_0.gguf", db_path="sqliterag.db"
            )

        self.settings = settings

        self._conn = sqlite3.connect(settings.db_path)

        self._repository = Repository(self._conn, settings)
        self._chunker = Chunker(self._conn, settings)
        self._engine = Engine(self._conn, settings, chunker=self._chunker)

        self.ready = False

    def _ensure_initialized(self):
        if not self.ready:
            Database.initialize(self._conn, self.settings)
            self._engine.load_model()

        self.ready = True

    def add(self, path: str, recursively: bool = False):
        """Add the file content into the database"""
        self._ensure_initialized()

        if not Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist.")

        files_to_process: list[Path] = []
        path_obj = Path(path)

        if path_obj.is_file():
            files_to_process.append(path_obj)
        elif path_obj.is_dir():
            if recursively:
                files_to_process = list(path_obj.rglob("*"))
            else:
                files_to_process = list(path_obj.glob("*"))

            files_to_process = [
                f
                for f in files_to_process
                if f.is_file() and FileReader.is_supported(f)
            ]

        for file_path in files_to_process:
            # TODO: check the file extension
            content = FileReader.parse_file(file_path)
            # TODO: include metadata extraction and mdx options (see our docsearch)
            document = Document(content=content, uri=str(file_path))
            chunks = self._chunker.chunk(document.content)
            chunks = self._engine.generate_embedding(chunks)
            document.chunks = chunks

            self._repository.add_document(document)

        if self.settings.quantize_scan:
            self._engine.quantize()

    def add_text(
        self, text: str, uri: Optional[str] = None, metadata: dict = {}
    ) -> None:
        """Add a text content into the database"""
        self._ensure_initialized()

        document = Document(content=text, uri=uri, metadata=metadata)
        chunks = self._chunker.chunk(document.content)
        chunks = self._engine.generate_embedding(chunks)
        document.chunks = chunks

        self._repository.add_document(document)

        if self.settings.quantize_scan:
            self._engine.quantize()

    def list_documents(self) -> list[Document]:
        """List all documents in the database"""
        self._ensure_initialized()

        return self._repository.list_documents()

    def find_document(self, identifier: str) -> Document | None:
        """Find document by ID or URI"""
        self._ensure_initialized()
        return self._repository.find_document_by_id_or_uri(identifier)

    def remove_document(self, identifier: str) -> bool:
        """Remove document by ID or URI"""
        self._ensure_initialized()
        
        # First find the document to get its ID
        document = self._repository.find_document_by_id_or_uri(identifier)
        if not document:
            return False
        
        return self._repository.remove_document(document.id or "")

    # def search(
    #     self, query: str, top_k: int = 10
    # ) -> list[Document]:
    #     """Search for documents matching the query"""
    #     self._ensure_initialized()