import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from sqlite_rag.logger import Logger
from sqlite_rag.models.document_result import DocumentResult

from .chunker import Chunker
from .database import Database
from .engine import Engine
from .models.document import Document
from .reader import FileReader
from .repository import Repository
from .settings import Settings, SettingsManager


class SQLiteRag:
    def __init__(self, connection: sqlite3.Connection, settings: Settings):
        self._settings = settings
        self._logger = Logger()

        self._conn = connection

        self._repository = Repository(self._conn, settings)
        self._chunker = Chunker(self._conn, settings)
        self._engine = Engine(self._conn, settings, chunker=self._chunker)

        self.ready = False

    def _ensure_initialized(self):
        if not self.ready:
            self._engine.load_model()

        self.ready = True

    @staticmethod
    def create(
        db_path: str = "./sqliterag.sqlite",
        settings: Optional[dict[str, Any]] = None,
        require_existing: bool = False,
    ) -> "SQLiteRag":
        """Create a new SQLiteRag instance with the given settings.

        It initializes the database connection and prepares the environment.
        If no new settings are provided, it uses the default settings or load
        the settings used in the last execution.

        Args:
            db_path: Path to the SQLite database file
            settings: Optional settings to override defaults
            require_existing: If True, raises FileNotFoundError if database doesn't exist
        """

        if require_existing and not Path(db_path).exists():
            raise FileNotFoundError(f"Database file {db_path} does not exist.")

        conn = Database.new_connection(db_path)

        settings_manager = SettingsManager(conn)
        current_settings = settings_manager.configure(settings)

        Database.initialize(conn, current_settings)

        return SQLiteRag(conn, current_settings)

    def add(
        self,
        path: str,
        recursive: bool = False,
        use_relative_paths: bool = False,
        metadata: dict = {},
    ) -> int:
        """Add the file content into the database"""
        self._ensure_initialized()

        if not Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist.")

        parent = Path(path).parent

        files_to_process = FileReader.collect_files(Path(path), recursive=recursive)

        self._engine.create_new_context()

        processed = 0
        total_to_process = len(files_to_process)
        self._logger.info(f"Processing {total_to_process} files...")
        try:
            for i, file_path in enumerate(files_to_process):
                content = FileReader.parse_file(
                    file_path, self._settings.max_document_size_bytes
                )

                if not content:
                    self._logger.warning(
                        f"{i+1}/{total_to_process} Skipping empty file: {file_path}"
                    )
                    continue

                uri = (
                    str(file_path.relative_to(parent))
                    if use_relative_paths
                    else str(file_path.absolute())
                )
                document = Document(content=content, uri=uri, metadata=metadata)

                exists = self._repository.document_exists_by_hash(document.hash())
                if exists:
                    self._logger.info(
                        f"{i+1}/{total_to_process} Unchanged: {file_path}"
                    )
                    continue

                self._logger.info(f"{i+1}/{total_to_process} Processing: {file_path}")
                document = self._engine.process(document)

                self._repository.add_document(document)

                processed += 1
        finally:
            if self._settings.quantize_scan:
                self._engine.quantize()

        self._engine.free_context()

        return processed

    def add_text(
        self, text: str, uri: Optional[str] = None, metadata: dict = {}
    ) -> None:
        """Add a text content into the database"""
        self._ensure_initialized()

        document = Document(content=text, uri=uri, metadata=metadata)

        self._engine.create_new_context()
        document = self._engine.process(document)

        self._repository.add_document(document)

        if self._settings.quantize_scan:
            self._engine.quantize()

        self._engine.free_context()

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

        self._engine.create_new_context()

        for i, doc in enumerate(documents):
            doc_id = doc.id or ""

            if doc.uri and Path(doc.uri).exists():
                # File still exists, recreate embeddings
                try:
                    content = FileReader.parse_file(
                        Path(doc.uri), self._settings.max_document_size_bytes
                    )
                    doc.content = content

                    self._repository.remove_document(doc_id)
                    processed_doc = self._engine.process(doc)
                    self._repository.add_document(processed_doc)

                    reprocessed += 1
                    self._logger.debug(f"{i+1}/{total_docs} Reprocessed: {doc.uri}")
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
                    self._logger.info(
                        f"{i+1}/{total_docs} Removed missing document: {doc.uri}"
                    )
            else:
                # Document without URI (text content)
                try:
                    self._repository.remove_document(doc_id)
                    processed_doc = self._engine.process(doc)
                    self._repository.add_document(processed_doc)

                    reprocessed += 1
                    self._logger.debug(
                        f"{i+1}/{total_docs} Reprocessed text document: {doc.content[:20]!r}..."
                    )
                except Exception as e:
                    self._logger.error(f"Error processing text document {doc.id}: {e}")

            if self._settings.quantize_scan:
                self._engine.quantize()

        self._engine.free_context()

        return {
            "total": total_docs,
            "reprocessed": reprocessed,
            "not_found": not_found,
            "removed": removed,
        }

    def reset(self) -> bool:
        """Reset/clear the entire database by deleting and recreating it"""
        db_path = self._conn.execute("PRAGMA database_list;").fetchone()[2]

        try:
            # Close the database connection
            self._conn.close()

            # Delete the database file if it exists
            if Path(db_path).exists():
                Path(db_path).unlink()
                self._logger.info(f"Deleted database file: {db_path}")

            return True

        except Exception as e:
            self._logger.error(f"Error during database reset: {e}")
            return False

    def search(
        self, query: str, top_k: int = 10, new_context: bool = True
    ) -> list[DocumentResult]:
        """Search for documents matching the query.

        Args:
            query: The search query string
            top_k: Number of top results to search in both semantic and FTS search.
                Number of documents may be higher.
            new_context: Whether to create a new LLM context for this search
        """
        self._ensure_initialized()
        if new_context:
            self._engine.create_new_context()

        if self._settings.use_prompt_templates:
            query = self._settings.prompt_template_retrieval_query.format(content=query)

        return self._engine.search(query, top_k=top_k)

    def get_settings(self) -> dict:
        """Get settings and more useful information"""
        versions = self._engine.versions()
        return {**versions, **asdict(self._settings)}

    def quantize_vectors(self) -> None:
        """Quantize vectors for faster search"""
        self._ensure_initialized()
        self._engine.quantize()

    def quantize_cleanup(self) -> None:
        """Clean up quantization structures"""
        self._ensure_initialized()
        self._engine.quantize_cleanup()

    def close(self) -> None:
        """Free up resources"""
        self._engine.close()
        if self._conn:
            self._conn.close()

    def __del__(self):
        self.close()
