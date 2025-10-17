import random
import string
from sqlite3 import OperationalError

import pytest

from sqlite_rag.chunker import Chunker
from sqlite_rag.engine import Engine
from sqlite_rag.models.document import Document
from sqlite_rag.repository import Repository
from sqlite_rag.sentence_splitter import SentenceSplitter


class TestEngine:
    @pytest.mark.slow
    def test_stress_embedding_generation(self, engine):
        """Test embedding generation with a large number of chunks
        to not fail and to never generate duplicated embeddings."""

        def random_string(length=30):
            return "".join(
                random.choices(string.ascii_letters + string.digits + " ", k=length)
            )

        result_chunks = {}
        for i in range(1000):
            try:
                embedding = engine.generate_embedding(random_string())
                result_chunks[embedding.hex()] = embedding
                assert len(result_chunks) == i + 1
            except Exception as e:
                pytest.fail(f"Embedding generation failed on chunk {i}: {e}")

        # Assert
        assert len(result_chunks) == 1000


class TestEngineQuantization:
    def test_quantize_embedding(self, engine):
        """Test quantize called for chunks and sentences embeddings."""
        engine.quantize()

        # If no exception is raised, the test passes
        engine.search("test query", "test query")

    def test_quantize_cleanup(self, engine):
        """Test quantize cleanup works without errors."""
        engine.quantize()
        engine.quantize_cleanup()

        with pytest.raises(OperationalError) as exc_info:
            engine.search("test query", "test query")
        assert "Ensure that vector_quantize() has been called" in str(exc_info.value)


class TestEngineSearch:
    def test_search_with_empty_database(self, engine):
        results = engine.search("nonexistent query", top_k=5)

        assert len(results) == 0

    def test_search_with_semantic_and_fts(self, db_conn):
        # Arrange
        conn, settings = db_conn

        engine = Engine(conn, settings, Chunker(conn, settings), SentenceSplitter())
        engine.load_model()
        engine.create_new_context()

        doc1 = Document(
            content="The quick brown fox jumps over the lazy dog.",
            uri="document1.txt",
        )
        doc2 = Document(
            content="How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            uri="document2.txt",
        )
        doc3 = Document(
            content="This document discusses about woodcutters and wood.",
            uri="document3.txt",
        )

        engine.process(doc1)
        engine.process(doc2)
        engine.process(doc3)

        repository = Repository(conn, settings)
        repository.add_document(doc1)
        repository.add_document(doc2)
        doc3_id = repository.add_document(doc3)

        engine.quantize()

        # Act
        results = engine.search("wood lumberjack", "wood lumberjack", top_k=5)

        assert len(results) > 0
        assert doc3_id == results[0].document.id

    def test_search_semantic_result(self, db_conn):
        # Arrange
        conn, settings = db_conn

        engine = Engine(conn, settings, Chunker(conn, settings), SentenceSplitter())
        engine.load_model()
        engine.create_new_context()

        doc1 = Document(
            content="The quick brown fox jumps over the lazy dog.",
            uri="document1.txt",
        )
        doc2 = Document(
            content="How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            uri="document2.txt",
        )
        doc3 = Document(
            content="This document discusses about woodcutters and wood.",
            uri="document3.txt",
        )

        engine.process(doc1)
        engine.process(doc2)
        engine.process(doc3)

        repository = Repository(conn, settings)
        repository.add_document(doc1)
        repository.add_document(doc2)
        doc3_id = repository.add_document(doc3)

        engine.quantize()

        # Act
        results = engine.search("about lumberjack", "about lumberjack", top_k=5)

        assert len(results) > 0
        assert doc3_id == results[0].document.id

    def test_search_fts_results(self, db_conn):
        # Arrange
        conn, settings = db_conn

        engine = Engine(conn, settings, Chunker(conn, settings), SentenceSplitter())
        engine.load_model()
        engine.create_new_context()

        doc1 = Document(
            content="The quick brown fox jumps over the lazy dog.",
            uri="document1.txt",
        )
        doc2 = Document(
            content="How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            uri="document2.txt",
        )
        doc3 = Document(
            content="This document discusses about woodcutters and wood.",
            uri="document3.txt",
        )

        engine.process(doc1)
        engine.process(doc2)
        engine.process(doc3)

        repository = Repository(conn, settings)
        doc1_id = repository.add_document(doc1)
        repository.add_document(doc2)
        repository.add_document(doc3)

        engine.quantize()

        # Act
        results = engine.search("quick brown fox", "quick brown fox", top_k=5)

        assert len(results) > 0
        assert doc1_id == results[0].document.id
        assert results[0].fts_rank
        assert results[0].fts_rank == 1
        assert results[0].fts_score

    def test_search_without_quantization(self, db_conn):
        # Arrange
        conn, settings = db_conn
        settings.quantize_scan = False

        engine = Engine(conn, settings, Chunker(conn, settings), SentenceSplitter())
        engine.load_model()

        doc = Document(
            content="The quick brown fox jumps over the lazy dog.",
            uri="document1.txt",
        )

        engine.create_new_context()
        engine.process(doc)

        repository = Repository(conn, settings)
        doc_id = repository.add_document(doc)

        # Act
        results = engine.search("wood lumberjack", "wood lumberjack")

        assert len(results) > 0
        assert doc_id == results[0].document.id

    def test_search_exact_match(self, db_conn):
        conn, settings = db_conn
        # cosin distance for searching embedding is exact 0.0 when strings match
        settings.other_vector_options = "distance=cosine"
        settings.use_prompt_templates = False

        engine = Engine(conn, settings, Chunker(conn, settings), SentenceSplitter())
        engine.load_model()
        engine.create_new_context()

        doc1 = Document(
            content="The quick brown fox jumps over the lazy dog",
            uri="document1.txt",
        )
        doc2 = Document(
            content="How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            uri="document2.txt",
        )

        engine.process(doc1)
        engine.process(doc2)

        repository = Repository(conn, settings)
        doc1_id = repository.add_document(doc1)
        repository.add_document(doc2)

        engine.quantize()

        # Act
        results = engine.search(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )

        assert len(results) > 0
        assert doc1_id == results[0].document.id
        assert 0.0 == results[0].vec_distance


class TestEngineSearchSentences:
    def test_search_sentences(self, db_conn):
        conn, settings = db_conn
        settings.use_prompt_templates = False
        settings.quantize_scan = False

        engine = Engine(conn, settings, Chunker(conn, settings), SentenceSplitter())
        engine.load_model()
        engine.create_new_context()

        doc = Document(
            content=(
                """The quick brown fox jumps over the lazy dog.
                A stitch in time saves nine.
                An apple a day keeps the doctor away.
                """
            ),
            uri="document1.txt",
        )

        engine.process(doc)

        repository = Repository(conn, settings)
        doc_id = repository.add_document(doc)

        cursor = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
        chunk_id = cursor.fetchone()[0]

        # Act
        results = engine.search_sentences(
            "stitch time",
            chunk_id,
            top_k=1,
        )

        assert len(results) > 0
        assert results[0].start_offset == 61  # it's the second sentence
        assert results[0].end_offset == 89
