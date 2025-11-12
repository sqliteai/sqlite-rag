import json
import re
import sqlite3
from typing import List

from sqlite_rag.errors import ContextSizeExceededError
from sqlite_rag.logger import Logger, LogLevel
from sqlite_rag.models.document_result import DocumentResult
from sqlite_rag.models.llm_model import LLMModel
from sqlite_rag.models.sentence_result import SentenceResult
from sqlite_rag.sentence_splitter import SentenceSplitter

from .chunker import Chunker
from .models.document import Document
from .settings import Settings


class Engine:
    # Considered a good default to normilize the score for RRF
    DEFAULT_RRF_K = 60

    def __init__(
        self,
        embedding_model: LLMModel,
        text_generation_model: LLMModel,
        settings: Settings,
        chunker: Chunker,
        sentence_splitter: SentenceSplitter,
    ):
        self._embedding_model = embedding_model
        self._text_generation_model = text_generation_model
        self._settings = settings
        self._chunker = chunker
        self._sentence_splitter = sentence_splitter
        self._logger = Logger(LogLevel.DEBUG)

    def process(self, document: Document) -> Document:
        if not document.get_title():
            document.set_generated_title()

        chunks = self._chunker.chunk(document)

        if self._settings.max_chunks_per_document > 0:
            chunks = chunks[: self._settings.max_chunks_per_document]

        for chunk in chunks:
            chunk.title = document.get_title()
            chunk.embedding = self.generate_embedding(chunk.get_embedding_text())

            sentences = self._sentence_splitter.split(chunk)
            for sentence in sentences:
                sentence.embedding = self.generate_embedding(sentence.content)
            chunk.sentences = sentences

        document.chunks = chunks

        return document

    def generate_embedding(self, text: str) -> bytes:
        """Generate embedding for the given text."""
        conn = self._embedding_model.ensure_loaded()

        try:
            cursor = conn.execute("SELECT llm_embed_generate(?) AS embedding", (text,))
        except sqlite3.Error as e:
            print(f"Error generating embedding for text\n: ```{text}```")
            raise e

        result = cursor.fetchone()

        if result is None:
            raise RuntimeError("Failed to generate embedding.")

        return result["embedding"]

    def quantize(self) -> None:
        """Quantize stored vector for faster search via quantized scan."""
        conn = self._embedding_model.ensure_loaded()
        cursor = conn.cursor()

        cursor.execute("SELECT vector_quantize('chunks', 'embedding');")
        cursor.execute("SELECT vector_quantize('sentences', 'embedding');")

        conn.commit()
        self._logger.debug("Quantization completed.")

    def quantize_preload(self) -> None:
        """Preload quantized vectors into memory for faster search."""
        cursor = self._embedding_model.ensure_loaded().cursor()

        cursor.execute("SELECT vector_quantize_preload('chunks', 'embedding');")
        cursor.execute("SELECT vector_quantize_preload('sentences', 'embedding');")

    def quantize_cleanup(self) -> None:
        """Clean up internal structures related to a previously quantized table/column."""
        conn = self._embedding_model.ensure_loaded()
        cursor = conn.cursor()

        cursor.execute("SELECT vector_quantize_cleanup('chunks', 'embedding');")
        cursor.execute("SELECT vector_quantize_cleanup('sentences', 'embedding');")

        conn.commit()

    def free_context(self) -> None:
        """Release resources associated with the current context."""
        cursor = self._embedding_model.ensure_loaded().cursor()

        cursor.execute("SELECT llm_context_free();")

    def search(self, query, top_k: int = 10) -> list[DocumentResult]:
        """Semantic search and full-text search sorted with Reciprocal Rank Fusion
        with top matching sentences to highlight."""
        semantic_query = query
        if self._settings.use_prompt_templates:
            semantic_query = self._settings.prompt_template_retrieval_query.format(
                content=query
            )

        # Clean up and split into words
        # '*' is used to match while typing
        fts_query = " ".join(re.findall(r"\b\w+\b", query.lower())) + "*"

        query_embedding = self.generate_embedding(semantic_query)

        results = self.search_documents(query_embedding, fts_query, top_k=top_k)

        # Refine chunks with top sentences
        for result in results:
            result.sentences = self.search_sentences(
                query_embedding, result.chunk_id, top_k=self._settings.top_k_sentences
            )

        return results

    def search_documents(
        self, query_embedding: bytes, fts_query: str, top_k: int
    ) -> list[DocumentResult]:
        """Semantic search and full-text search sorted with Reciprocal Rank Fusion."""
        # invalid query
        if query_embedding == b"" or fts_query.strip() == "":
            return []

        vector_scan_type = (
            "vector_quantize_scan"
            if self._settings.quantize_scan
            else "vector_full_scan"
        )

        cursor = self._embedding_model.ensure_loaded().cursor()

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
                chunks.id AS chunk_id,
                chunks.content AS chunk_content,
                vec_rank,
                fts_rank,
                combined_rank,
                vec_distance,
                fts_score
            FROM matches
                JOIN chunks ON chunks.id = matches.chunk_id
                JOIN documents ON documents.id = chunks.document_id
            ORDER BY combined_rank DESC
            LIMIT :k
            ;
            """,  # nosec B608
            {
                "query": fts_query,
                "query_embedding": query_embedding,
                "k": top_k,
                "rrf_k": Engine.DEFAULT_RRF_K,
                "weight_fts": self._settings.weight_fts,
                "weight_vec": self._settings.weight_vec,
            },
        )

        rows = cursor.fetchall()
        results = [
            DocumentResult(
                document=Document(
                    id=row["id"],
                    uri=row["uri"],
                    content=row["document_content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                ),
                chunk_id=row["chunk_id"],
                chunk_content=row["chunk_content"],
                vec_rank=row["vec_rank"],
                fts_rank=row["fts_rank"],
                combined_rank=row["combined_rank"],
                vec_distance=row["vec_distance"],
                fts_score=row["fts_score"],
            )
            for row in rows
        ]

        return results

    def search_sentences(
        self, query_embedding: bytes, chunk_id: int, top_k: int
    ) -> List[SentenceResult]:
        """Semantic search for sentences within a chunk."""
        vector_scan_type = (
            "vector_quantize_scan_stream"
            if self._settings.quantize_scan
            else "vector_full_scan_stream"
        )

        conn = self._embedding_model.ensure_loaded()

        cursor = conn.execute(
            f"""
            WITH vec_matches AS (
                SELECT
                    v.rowid AS sentence_id,
                    row_number() OVER (ORDER BY v.distance) AS rank_number,
                    v.distance
                FROM {vector_scan_type}('sentences', 'embedding', :query_embedding) AS v
                    JOIN sentences ON sentences.rowid = v.rowid
                WHERE sentences.chunk_id = :chunk_id
                LIMIT :top_k
            )
            SELECT
                sentence_id,
                -- Extract sentence directly from document content
                COALESCE(
                    substr(chunks.content, sentences.start_offset + 1, sentences.end_offset - sentences.start_offset),
                    ""
                ) AS content,
                sentences.start_offset AS sentence_start_offset,
                sentences.end_offset AS sentence_end_offset,
                rank_number,
                distance
            FROM vec_matches
                JOIN sentences ON sentences.rowid = vec_matches.sentence_id
                JOIN chunks ON chunks.id = sentences.chunk_id
            ORDER BY rank_number ASC
            """,  # nosec B608
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "chunk_id": chunk_id,
            },
        )

        rows = cursor.fetchall()
        sentences = []
        for row in rows:
            sentences.append(
                SentenceResult(
                    id=row["sentence_id"],
                    chunk_id=chunk_id,
                    content=row["content"].strip(),
                    rank=row["rank_number"],
                    distance=row["distance"],
                    start_offset=row["sentence_start_offset"],
                    end_offset=row["sentence_end_offset"],
                )
            )

        return sentences[:top_k]

    def create_new_chat(self) -> None:
        """Create a new LLM chat context with empty history."""
        conn = self._text_generation_model.ensure_loaded()

        conn.execute(
            "SELECT llm_context_create(?);",
            (self._settings.get_context_options_text_generation(),),
        )

        conn.execute("SELECT llm_chat_create();")

    def ask(self, query: str) -> sqlite3.Cursor:
        """Generate an answer to the query using the LLM."""
        results = self.search(query, top_k=3)

        context = ""
        for result in results:
            self._logger.debug(
                f"doc uri: {result.document.uri}, vector: {result.vec_distance}, fts: {result.fts_score}, score: {result.combined_rank}"
            )
            if result.combined_rank > self._settings.results_threshold:
                self._logger.debug("\r\b - taken")
                # TODO: how to improve context limit?
                preview = result.document.content[:5000].replace("\n", "\\n")
                context += f"{preview}\n\n"

        prompt = query
        if context != "":
            # prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say you that don't know. Use three sentences maximum and keep the answer coincise.
            # prompt = prompt = f"""Answer the question based only on the following documents. Answer with the summary of the documents provided. Do **NOT** include any introductory phrases, titles, or prefixes such as "Answer:", "The answer is:", "Final Answer:", or "Based on the context,". Start your response with the answer itself:"""
            prompt = f"""Answer the question based on the following documents.
Answer with the summary of the documents provided.
Do **NOT** include any introductory phrases, titles, or prefixes such as "Answer:", "The answer is:", "Final Answer:", or "Based on the context,". Start your response with the answer itself:

{context}

{query}
"""
        conn = self._text_generation_model.ensure_loaded()

        # TODO: token count is duplicated in Chunker class
        context_size = conn.execute("SELECT llm_token_count(?);", (prompt,)).fetchone()[
            0
        ]

        if context_size > self._settings.context_size_text_gen:
            raise ContextSizeExceededError(
                f"Prompt size ({context_size} tokens) exceeds context size limit ({self._settings.context_size_text_gen} tokens). The model may not be able to process the entire prompt."
            )

        conn.execute("SELECT llm_sampler_init_temp(?);", (self._settings.temp,))
        conn.execute("SELECT llm_sampler_init_top_k(?);", (self._settings.top_k,))
        conn.execute(
            "SELECT llm_sampler_init_top_p(?, ?);",
            (self._settings.top_p, self._settings.top_p_min_keep),
        )
        conn.execute(
            "SELECT llm_sampler_init_min_p(?, ?);",
            (self._settings.min_p, self._settings.min_p_min_keep),
        )
        conn.execute(
            "SELECT llm_sampler_init_penalties(?, ?, ?, ?);",
            (
                self._settings.penaltiy_n_tokens,
                self._settings.penalty_repeat,
                self._settings.penalty_frequency,
                self._settings.penalty_presence,
            ),
        )
        conn.execute("SELECT llm_sampler_init_dist(?);", (self._settings.random_seed,))

        # With the cursor response can be streamed by fetching single rows
        cursor = conn.execute("SELECT reply FROM llm_chat(?);", (prompt,))

        return cursor

    def versions(self) -> dict:
        """Get versions of the loaded extensions."""
        conn = self._embedding_model.ensure_loaded()
        cursor = conn.execute(
            "SELECT ai_version() AS ai_version, vector_version() AS vector_version;"
        )
        row = cursor.fetchone()

        return {
            "ai_version": row["ai_version"],
            "vector_version": row["vector_version"],
        }
