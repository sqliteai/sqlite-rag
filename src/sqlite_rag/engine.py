import sqlite3
from pathlib import Path

from .chunker import Chunker
from .models.chunk import Chunk
from .models.document import Document
from .settings import Settings


class Engine:
    def __init__(self, conn: sqlite3.Connection, settings: Settings, chunker: Chunker):
        self._conn = conn
        self.settings = settings
        self._chunker = chunker

    def load_model(self):
        """Load the model model from the specified path
        or download it from Hugging Face if not found."""

        model_path = Path(self.settings.model_path_or_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # model_path = self.settings.model_path_or_name
        # if not Path(self.settings.model_path_or_name).exists():
        #     # check if exists locally or try to download it from Hugging Face
        #     model_path = hf_hub_download(
        #         repo_id=self.settings.model_path_or_name,
        #         filename="model-q4_0.gguf",  # GGUF format
        #         cache_dir="./models"
        #     )

        self._conn.execute(
            f"SELECT llm_model_load('{self.settings.model_path_or_name}', '{self.settings.model_config}');"
        )

    def process(self, document: Document) -> Document:
        chunks = self._chunker.chunk(document.content)
        chunks = self.generate_embedding(chunks)
        document.chunks = chunks
        return document

    # TODO: better to get a list of str and return a list of embeddings?
    def generate_embedding(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embedding for the given text."""
        cursor = self._conn.cursor()

        for chunk in chunks:
            cursor.execute("SELECT llm_embed_generate(?)", (chunk.content,))
            result = cursor.fetchone()

            if result is None:
                raise RuntimeError("Failed to generate embedding.")

            chunk.embedding = result[0]

        return chunks

    def quantize(self) -> None:
        """Quantitize stored vector for faster search via quantitize scan."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT vector_quantize('chunks', 'embedding');")

        self._conn.commit()

    def search(self, query: str, limit: int = 10) -> list[Document]:
        """Semantic search and full-text search sorted with Reciprocal Rank Fusion."""
        cursor = self._conn.cursor()

        query_embedding = self.generate_embedding([Chunk(content=query)])[0].embedding

        cursor.execute(
            # TODO: use vector_convert_XXX to convert the query to the correct type
            """
            -- sqlite-vector KNN vector search results
            WITH vec_matches AS (
                SELECT
                    row_number() over (order by v.distance) AS rank_number,
                    c.document_id,
                    v.distance
                FROM chunks AS c
                    JOIN vector_quantize_scan('chunks', 'embedding', vector_convert_f32(:query_embedding), :k) AS v
                    ON c.rowid = v.rowid
            ),
            -- Full-text search results
            fts_matches AS (
                SELECT
                    row_number() over (order by rank) AS rank_number,
                    c.document_id,
                    rank AS score
                FROM chunks_fts AS c_fts
                    JOIN chunks AS c ON c_fts.rowid = c.rowid
                WHERE c_fts.content MATCH :query
                LIMIT :k
            ),
            -- combine FTS5 + vector search results with RRF
            matches AS (
                SELECT
                    documents.id,
                    documents.uri,
                    documents.content,
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
                        ON vec_matches.document_id = fts_matches.document_id
                    JOIN documents ON documents.id = COALESCE(vec_matches.document_id, fts_matches.document_id)
                ORDER BY combined_rank DESC
            )
            SELECT
                id,
                uri,
                content,
                vec_rank,
                fts_rank,
                combined_rank,
                vec_distance,
                fts_score
            FROM matches;
            """,
            {
                "query": query + "*",
                "query_embedding": query_embedding,
                "k": limit,
                # TODO: move to settings or costants
                "rrf_k": 60,
                "weight_fts": 1.0,
                "weight_vec": 1.0,
            },
        )

        rows = cursor.fetchall()
        return [
            Document(
                id=row[0],
                uri=row[1],
                content=row[2],
                vec_rank=row[3],
                fts_rank=row[4],
                combined_rank=row[5],
                vec_distance=row[6],
                fts_score=row[7],
            )
            for row in rows
        ]
