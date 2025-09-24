import pytest

from sqlite_rag.chunker import Chunker
from sqlite_rag.engine import Engine
from sqlite_rag.models.chunk import Chunk
from sqlite_rag.models.document import Document
from sqlite_rag.repository import Repository
from sqlite_rag.settings import Settings


class TestEngine:
    def test_generate_embedding(self, engine):
        chunk = Chunk(content="This is a test chunk for embedding generation.")

        result_chunks = engine.generate_embedding([chunk])

        assert len(result_chunks) == 1
        assert result_chunks[0].embedding is not None
        assert isinstance(result_chunks[0].embedding, bytes)

    @pytest.mark.parametrize("use_prompt_templates", [True, False])
    def test_generate_embedding_with_prompt_template(
        self, mocker, use_prompt_templates
    ):
        # Arrange
        mock_conn = mocker.Mock()
        mock_cursor = mocker.Mock()
        mock_cursor.fetchone.return_value = {"embedding": b"fake_embedding"}
        mock_conn.cursor.return_value = mock_cursor

        settings = Settings(
            use_prompt_templates=use_prompt_templates,
            prompt_template_retrieval_document="Title: {title}\nContent: {content}",
        )

        engine = Engine(mock_conn, settings, mocker.Mock())

        chunk = Chunk(
            content="Test content",
            title="Test Title",
        )

        # Act
        engine.generate_embedding([chunk])

        # Assert - verify cursor.execute was called with formatted template
        expected_content = (
            "Title: Test Title\nContent: Test content"
            if use_prompt_templates
            else "Test content"
        )
        mock_cursor.execute.assert_called_with(
            "SELECT llm_embed_generate(?) AS embedding", (expected_content,)
        )

    def test_extract_document_title(self):
        text = """# This is the Title
        This is the content of the document.
        It has multiple lines.
        """

        engine = Engine(None, Settings(), None)  # type: ignore

        title = engine.extract_document_title(text)
        assert title == "This is the Title"

    @pytest.mark.parametrize(
        "fallback, expected_title",
        [
            (True, "This is the first line of the document without a title."),
            (False, None),
        ],
    )
    def test_extract_document_title_from_first_line(self, fallback, expected_title):
        text = """
        This is the first line of the document without a title.
        It has multiple lines.
        """

        engine = Engine(None, Settings(), None)  # type: ignore

        title = engine.extract_document_title(text, fallback)
        assert title == expected_title

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
