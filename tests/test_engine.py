from sqlite_rag.chunker import Chunker
from sqlite_rag.engine import Engine
from sqlite_rag.models.chunk import Chunk
from sqlite_rag.models.document import Document
from sqlite_rag.repository import Repository


class TestEngine:
    def test_generate_embedding(self, engine):
        chunk = Chunk(content="This is a test chunk for embedding generation.")

        result_chunks = engine.generate_embedding([chunk])

        assert len(result_chunks) == 1
        assert result_chunks[0].embedding is not None
        assert isinstance(result_chunks[0].embedding, bytes)

    def test_search_with_empty_database(self, engine):
        results = engine.search("nonexistent query", limit=5)

        assert len(results) == 0

    def test_search_with_semantic_and_fts(self, db_conn):
        # Arrange
        conn, settings = db_conn

        engine = Engine(conn, settings, Chunker(conn, settings))
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
        results = engine.search("wood lumberjack", limit=5)

        assert len(results) > 0
        assert doc3_id == results[0].document.id

    def test_search_semantic_result(self, db_conn):
        # Arrange
        conn, settings = db_conn

        engine = Engine(conn, settings, Chunker(conn, settings))
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
        results = engine.search("about lumberjack", limit=5)

        assert len(results) > 0
        assert doc3_id == results[0].document.id

    def test_search_fts_results(self, db_conn):
        # Arrange
        conn, settings = db_conn

        engine = Engine(conn, settings, Chunker(conn, settings))
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
        results = engine.search("quick brown fox", limit=5)

        assert len(results) > 0
        assert doc1_id == results[0].document.id

    def test_search_without_quantization(self, db_conn):
        # Arrange
        conn, settings = db_conn
        settings.quantize_scan = False

        engine = Engine(conn, settings, Chunker(conn, settings))
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
        results = engine.search("wood lumberjack")

        assert len(results) > 0
        assert doc_id == results[0].document.id

    def test_search_exact_match(self, db_conn):
        conn, settings = db_conn
        # cosin distance for searching embedding is exact 0.0 when strings match
        settings.other_vector_options = "distance=cosine"

        engine = Engine(conn, settings, Chunker(conn, settings))
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
        results = engine.search("The quick brown fox jumps over the lazy dog")

        assert len(results) > 0
        assert doc1_id == results[0].document.id
        assert 0.0 == results[0].vec_distance
