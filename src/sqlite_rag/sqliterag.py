import sqlite3
from pathlib import Path
from typing import Optional

from sqlite_rag.logger import Logger
from sqlite_rag.models.document_result import DocumentResult

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
                model_path_or_name="./Qwen3-Embedding-0.6B-Q8_0.gguf",
                db_path="sqliterag.db",
            )

        self.settings = settings
        self._logger = Logger()

        self._conn = self._create_db_connection()

        self._repository = Repository(self._conn, settings)
        self._chunker = Chunker(self._conn, settings)
        self._engine = Engine(self._conn, settings, chunker=self._chunker)

        self.ready = False

    def _create_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.settings.db_path)
        conn.row_factory = sqlite3.Row

        return conn

    def _ensure_initialized(self):
        if not self.ready:
            Database.initialize(self._conn, self.settings)
            self._engine.load_model()

        self.ready = True

    def add(
        self,
        path: str,
        recursive: bool = False,
        absolute_paths: bool = True,
        metadata: dict = {},
    ) -> int:
        """Add the file content into the database"""
        self._ensure_initialized()

        if not Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist.")

        parent = Path(path).parent

        files_to_process = FileReader.collect_files(Path(path), recursive=recursive)

        self._logger.info(f"Processing {len(files_to_process)} files...")
        for file_path in files_to_process:
            content = FileReader.parse_file(file_path)

            uri = (
                str(file_path.absolute())
                if absolute_paths
                else str(file_path.relative_to(parent))
            )
            document = Document(content=content, uri=uri, metadata=metadata)

            exists = self._repository.document_exists_by_hash(document.hash())
            if exists:
                self._logger.info(f"Unchanged: {file_path}")
                continue

            self._logger.info(f"Processing: {file_path}")
            document = self._engine.process(document)

            self._repository.add_document(document)

        # TODO: when is it better to quantize? after each document?
        if self.settings.quantize_scan:
            self._engine.quantize()

        return len(files_to_process)

    def add_text(
        self, text: str, uri: Optional[str] = None, metadata: dict = {}
    ) -> None:
        """Add a text content into the database"""
        self._ensure_initialized()

        document = Document(content=text, uri=uri, metadata=metadata)
        document = self._engine.process(document)

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
        if not document or not document.id:
            return False

        return self._repository.remove_document(document.id)

    def rebuild(self, remove_missing: bool = False) -> dict:
        """Rebuild embeddings and full-text index for all documents"""
        self._ensure_initialized()

        documents = self._repository.list_documents()
        total_docs = len(documents)
        reprocessed = 0
        not_found = 0
        removed = 0

        for doc in documents:
            doc_id = doc.id or ""

            if doc.uri and Path(doc.uri).exists():
                # File still exists, recreate embeddings
                try:
                    content = FileReader.parse_file(Path(doc.uri))
                    doc.content = content

                    self._repository.remove_document(doc_id)
                    processed_doc = self._engine.process(doc)
                    self._repository.add_document(processed_doc)

                    reprocessed += 1
                    self._logger.debug(f"Reprocessed: {doc.uri}")
                except Exception as e:
                    self._logger.error(f"Error processing {doc.uri}: {e}")
                    not_found += 1
            elif doc.uri:
                # File not found
                not_found += 1
                self._logger.warning(f"File not found: {doc.uri}")

                if remove_missing:
                    self._repository.remove_document(doc.id or "")
                    removed += 1
                    self._logger.info(f"Removed missing document: {doc.uri}")
            else:
                # Document without URI (text content)
                try:
                    self._repository.remove_document(doc_id)
                    processed_doc = self._engine.process(doc)
                    self._repository.add_document(processed_doc)

                    reprocessed += 1
                    self._logger.debug(
                        f"Reprocessed text document: {doc.content[:20]!r}..."
                    )
                except Exception as e:
                    self._logger.error(f"Error processing text document {doc.id}: {e}")

        if self.settings.quantize_scan:
            self._engine.quantize()

        return {
            "total": total_docs,
            "reprocessed": reprocessed,
            "not_found": not_found,
            "removed": removed,
        }

    def reset(self) -> bool:
        """Reset/clear the entire database by deleting and recreating it"""
        db_path = self.settings.db_path

        try:
            # Close the database connection
            self._conn.close()

            # Delete the database file if it exists
            if Path(db_path).exists():
                Path(db_path).unlink()
                self._logger.info(f"Deleted database file: {db_path}")

            # Recreate the database connection and initialize
            self._conn = self._create_db_connection()

            # Reinitialize components with new connection
            self._repository = Repository(self._conn, self.settings)
            self._chunker = Chunker(self._conn, self.settings)
            self._engine = Engine(self._conn, self.settings, chunker=self._chunker)

            # Reset ready flag so initialization happens on next use
            self.ready = False

            return True

        except Exception as e:
            self._logger.error(f"Error during database reset: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> list[DocumentResult]:
        """Search for documents matching the query"""
        self._ensure_initialized()

        return self._engine.search(query, limit=top_k)
