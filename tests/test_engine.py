import pytest

from sqlite_rag.chunker import Chunker
from sqlite_rag.engine import Engine
from sqlite_rag.models.chunk import Chunk
from sqlite_rag.models.document import Document
from sqlite_rag.repository import Repository
from sqlite_rag.settings import Settings


class TestEngine:
    def test_generate_embedding(self, engine):
        text = "This is a test chunk for embedding generation."

        embedding = engine.generate_embedding(text)

        assert embedding is not None
        assert isinstance(embedding, bytes)

    def test_process_uses_get_embedding_text(self, mocker):
        settings = Settings("test-model", use_prompt_templates=True)
        settings.prompt_template_retrieval_document = "title: {title} | text: {content}"

        # Create a single mock chunk
        mock_chunk = Chunk(
            content="Test content",
            title="Test Doc",
            head_overlap_text="overlap text",
            prompt=settings.prompt_template_retrieval_document,
        )

        mock_conn = mocker.Mock()
        mock_chunker = mocker.Mock()
        mock_chunker.chunk.return_value = [mock_chunk]

        engine = Engine(mock_conn, settings, mock_chunker)

        # Mock generate_embedding completely
        mock_generate = mocker.patch.object(
            engine, "generate_embedding", return_value=b"mock_embedding"
        )

        document = Document(content="Some text", metadata={"title": "Test Doc"})
        engine.process(document)

        # Assert generate_embedding was called with chunk.get_embedding_text()
        expected_text = mock_chunk.get_embedding_text()
        mock_generate.assert_called_once_with(expected_text)

    @pytest.mark.parametrize(
        "max_chunks_per_document, expected_chunk_count",
        [(0, 2), (1, 1), (4, 2)],
    )
    def test_process_with_max_chunks_per_document(
        self, mocker, max_chunks_per_document, expected_chunk_count
    ):
        # Arrange
        chunks = [
            Chunk(content="Chunk 1"),
            Chunk(content="Chunk 2"),
            Chunk(content="Chunk 3"),
        ]

        mock_conn = mocker.Mock()
        settings = Settings(max_chunks_per_document=max_chunks_per_document)
        mock_chunker = mocker.Mock()
        mock_chunker.chunk.return_value = chunks

        engine = Engine(mock_conn, settings, mock_chunker)

        mock_generate_embedding = mocker.patch.object(engine, "generate_embedding")
        mock_generate_embedding = mocker.spy(
            mock_generate_embedding, "generate_embedding"
        )
        mock_generate_embedding.return_value = chunks

        document = Document(content="Test document content")

        # Act
        engine.process(document)

        # Assert
        for call_args in mock_generate_embedding.call_args_list:
            chunks = call_args[0][0]  # First argument
            assert len(chunks) == expected_chunk_count


class TestEngineSearch:
    def test_search_with_empty_database(self, engine):
        results = engine.search("nonexistent query", top_k=5)

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
        results = engine.search("wood lumberjack", top_k=5)

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
        results = engine.search("about lumberjack", top_k=5)

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
        results = engine.search("quick brown fox", top_k=5)

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
        settings.use_prompt_templates = False

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
