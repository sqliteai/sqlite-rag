import json
import re
import sqlite3
from pathlib import Path

from sqlite_rag.logger import Logger
from sqlite_rag.models.document_result import DocumentResult

from .chunker import Chunker
from .models.document import Document
from .settings import Settings


class Engine:
    # Considered a good default to normilize the score for RRF
    DEFAULT_RRF_K = 60

    def __init__(self, conn: sqlite3.Connection, settings: Settings, chunker: Chunker):
        self._conn = conn
        self._settings = settings
        self._chunker = chunker
        self._logger = Logger()

    def load_model(self):
        """Load the model model from the specified path."""

        model_path = Path(self._settings.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self._conn.execute(
            "SELECT llm_model_load(?, ?);",
            (self._settings.model_path, self._settings.model_options),
        )

    def process(self, document: Document) -> Document:
        if not document.get_title():
            document.set_generated_title()

        chunks = self._chunker.chunk(document)

        if self._settings.max_chunks_per_document > 0:
            chunks = chunks[: self._settings.max_chunks_per_document]

        for chunk in chunks:
            chunk.title = document.get_title()
            chunk.embedding = self.generate_embedding(chunk.get_embedding_text())

        document.chunks = chunks

        return document

    def generate_embedding(self, text: str) -> bytes:
        """Generate embedding for the given text."""
        cursor = self._conn.cursor()

        try:
            cursor.execute("SELECT llm_embed_generate(?) AS embedding", (text,))
        except sqlite3.Error as e:
            print(f"Error generating embedding for text\n: ```{text}```")
            raise e

        result = cursor.fetchone()

        if result is None:
            raise RuntimeError("Failed to generate embedding.")

        return result["embedding"]

    def quantize(self) -> None:
        """Quantize stored vector for faster search via quantized scan."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT vector_quantize('chunks', 'embedding');")

        self._conn.commit()
        self._logger.debug("Quantization completed.")

    def quantize_preload(self) -> None:
        """Preload quantized vectors into memory for faster search."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT vector_quantize_preload('chunks', 'embedding');")

    def quantize_cleanup(self) -> None:
        """Clean up internal structures related to a previously quantized table/column."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT vector_quantize_cleanup('chunks', 'embedding');")

        self._conn.commit()

    def create_new_context(self) -> None:
        """"""
        cursor = self._conn.cursor()

        cursor.execute(
            "SELECT llm_context_create(?);", (self._settings.model_context_options,)
        )

    def free_context(self) -> None:
        """Release resources associated with the current context."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT llm_context_free();")

    def search(self, query: str, top_k: int = 10) -> list[DocumentResult]:
        """Semantic search and full-text search sorted with Reciprocal Rank Fusion."""
        query_embedding = self.generate_embedding(query)

        # Clean up and split into words
        # '*' is used to match while typing
        query = " ".join(re.findall(r"\b\w+\b", query.lower())) + "*"

        vector_scan_type = (
            "vector_quantize_scan"
            if self._settings.quantize_scan
            else "vector_full_scan"
        )

        cursor = self._conn.cursor()
        # TODO: understand how to sort results depending on the distance metric
        # Eg, for cosine distance, higher is better (closer to 1)
        cursor.execute(
            f"""
            -- sqlite-vector KNN vector search results
            WITH vec_matches AS (
                SELECT
                    v.rowid AS chunk_id,
                    row_number() OVER (ORDER BY v.distance) AS rank_number,
                    v.distance
                FROM {vector_scan_type}('chunks', 'embedding', :query_embedding, :k) AS v
            ),
            -- Full-text search results
            fts_matches AS (
                SELECT
                    chunks_fts.rowid AS chunk_id,
                    row_number() OVER (ORDER BY rank) AS rank_number,
                    rank AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH :query
                LIMIT :k
            ),
            -- combine FTS5 + vector search results with RRF
            matches AS (
                SELECT
                    COALESCE(vec_matches.chunk_id, fts_matches.chunk_id) AS chunk_id,
                    vec_matches.rank_number AS vec_rank,
                    fts_matches.rank_number AS fts_rank,
                    -- Reciprocal Rank Fusion score
                    (
                        COALESCE(1.0 / (:rrf_k + vec_matches.rank_number), 0.0) * :weight_vec +
                        COALESCE(1.0 / (:rrf_k + fts_matches.rank_number), 0.0) * :weight_fts
                    ) AS combined_rank,
                    vec_matches.distance AS vec_distance,
                    fts_matches.score AS fts_score
                FROM vec_matches
                    FULL OUTER JOIN fts_matches
                        ON vec_matches.chunk_id = fts_matches.chunk_id
            )
            SELECT
                documents.id,
                documents.uri,
                documents.content as document_content,
                documents.metadata,
                chunks.content AS snippet,
                vec_rank,
                fts_rank,
                combined_rank,
                vec_distance,
                fts_score
            FROM matches
                JOIN chunks ON chunks.id = matches.chunk_id
                JOIN documents ON documents.id = chunks.document_id
            ORDER BY combined_rank DESC
            ;
            """,  # nosec B608
            {
                "query": query,
                "query_embedding": query_embedding,
                "k": top_k,
                "rrf_k": Engine.DEFAULT_RRF_K,
                "weight_fts": self._settings.weight_fts,
                "weight_vec": self._settings.weight_vec,
            },
        )

        rows = cursor.fetchall()
        return [
            DocumentResult(
                document=Document(
                    id=row["id"],
                    uri=row["uri"],
                    content=row["document_content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                ),
                snippet=row["snippet"],
                vec_rank=row["vec_rank"],
                fts_rank=row["fts_rank"],
                combined_rank=row["combined_rank"],
                vec_distance=row["vec_distance"],
                fts_score=row["fts_score"],
            )
            for row in rows
        ]

    def versions(self) -> dict:
        """Get versions of the loaded extensions."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT ai_version() AS ai_version, vector_version() AS vector_version;"
        )
        row = cursor.fetchone()

        return {
            "ai_version": row["ai_version"],
            "vector_version": row["vector_version"],
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.execute("SELECT llm_model_free();")
            except sqlite3.ProgrammingError:
                # When connection is already closed the model
                # is already freed.
                pass

    def __del__(self):
        self.close()
