import pytest

from sqlite_rag.engine import Engine
from sqlite_rag.models.chunk import Chunk
from sqlite_rag.models.document import Document
from sqlite_rag.models.sentence import Sentence
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
        mock_sentence_splitter = mocker.Mock()
        mock_sentence_splitter.split.return_value = []

        engine = Engine(mock_conn, settings, mock_chunker, mock_sentence_splitter)

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
        mock_sentence_splitter = mocker.Mock()
        mock_sentence_splitter.split.return_value = []

        engine = Engine(mock_conn, settings, mock_chunker, mock_sentence_splitter)

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

    def test_process_with_sentences(self, mocker):
        # Arrange
        chunks = [Chunk(content="Chunk 1"), Chunk(content="Chunk 2")]

        mock_conn = mocker.Mock()
        settings = Settings()
        mock_chunker = mocker.Mock()
        mock_chunker.chunk.return_value = chunks
        mock_sentence_splitter = mocker.Mock()
        # return different number of sentences per chunk
        mock_sentence_splitter.split.side_effect = [
            [Sentence(content="Sentence 1.1")],
            [Sentence(content="Sentence 2.1"), Sentence(content="Sentence 2.2")],
        ]

        engine = Engine(mock_conn, settings, mock_chunker, mock_sentence_splitter)

        mock_generate_embedding = mocker.patch.object(engine, "generate_embedding")
        mock_generate_embedding = mocker.spy(
            mock_generate_embedding, "generate_embedding"
        )
        mock_generate_embedding.return_value = chunks

        document = Document(content="Test document content")

        # Act
        engine.process(document)

        # Assert
        assert len(document.chunks) == 2
        assert len(document.chunks[0].sentences) == 1
        assert len(document.chunks[1].sentences) == 2

    def test_process_without_sentences(self, mocker):
        # Arrange
        chunks = [Chunk(content="Chunk 1")]

        mock_conn = mocker.Mock()
        settings = Settings()
        mock_chunker = mocker.Mock()
        mock_chunker.chunk.return_value = chunks
        mock_sentence_splitter = mocker.Mock()
        mock_sentence_splitter.split.return_value = []

        engine = Engine(mock_conn, settings, mock_chunker, mock_sentence_splitter)

        mock_generate_embedding = mocker.patch.object(engine, "generate_embedding")
        mock_generate_embedding = mocker.spy(
            mock_generate_embedding, "generate_embedding"
        )
        mock_generate_embedding.return_value = chunks

        document = Document(content="Test document content")

        # Act
        engine.process(document)

        # Assert
        assert len(document.chunks) == 1
        assert len(document.chunks[0].sentences) == 0

    def test_search(self, mocker):
        # Arrange
        mock_conn = mocker.Mock()
        settings = Settings()
        engine = Engine(mock_conn, settings, mocker.Mock(), mocker.Mock())

        mock_generate = mocker.patch.object(
            engine, "generate_embedding", return_value=b"embedding"
        )
        mock_search_docs = mocker.patch.object(
            engine,
            "search_documents",
            return_value=[
                mocker.Mock(chunk_id=1, sentences=[]),
                mocker.Mock(chunk_id=2, sentences=[]),
            ],
        )
        mock_search_sents = mocker.patch.object(
            engine, "search_sentences", return_value=[]
        )

        # Act
        engine.search("test query", top_k=5)

        # Assert
        mock_generate.assert_called_once_with('title: "none" | text: test query')
        mock_search_docs.assert_called_once_with(b"embedding", "test query*", top_k=5)
        assert mock_search_sents.call_count == 2
        mock_search_sents.assert_any_call(
            b"embedding", 1, top_k=settings.top_k_sentences
        )
        mock_search_sents.assert_any_call(
            b"embedding", 2, top_k=settings.top_k_sentences
        )

    def test_search_uses_retrieval_query_template(self, mocker):
        # Arrange
        template = "task: search | Do something with {content}"

        settings = Settings(prompt_template_retrieval_query=template)

        mock_conn = mocker.Mock()
        engine = Engine(mock_conn, settings, mocker.Mock(), mocker.Mock())

        mock_generate = mocker.patch.object(
            engine, "generate_embedding", return_value=b"embedding"
        )
        mock_search_docs = mocker.patch.object(
            engine,
            "search_documents",
            return_value=[
                mocker.Mock(chunk_id=1, sentences=[]),
            ],
        )
        mock_search_sents = mocker.patch.object(
            engine, "search_sentences", return_value=[]
        )

        # Act
        query = "test query"
        engine.search(query, top_k=10)

        expected_fts_query = query + "*"

        # Assert
        # Is called with the formatted template
        mock_generate.assert_called_once_with(
            "task: search | Do something with test query"
        )
        mock_search_docs.assert_called_once_with(
            b"embedding", expected_fts_query, top_k=10
        )
        mock_search_sents.assert_called_once_with(
            b"embedding", 1, top_k=settings.top_k_sentences
        )

    @pytest.mark.parametrize("use_prompt_templates", [True, False])
    def test_search_with_prompt_template(self, mocker, use_prompt_templates):
        # Arrange
        settings = Settings(
            use_prompt_templates=use_prompt_templates,
            prompt_template_retrieval_query="task: search result | query: {content}",
        )

        mock_conn = mocker.Mock()
        engine = Engine(mock_conn, settings, mocker.Mock(), mocker.Mock())

        mock_generate_embedding = mocker.patch.object(
            engine, "generate_embedding", return_value=b"embedding"
        )
        mocker.patch.object(
            engine,
            "search_documents",
            return_value=[
                mocker.Mock(chunk_id=1, sentences=[]),
            ],
        )
        mocker.patch.object(engine, "search_sentences", return_value=[])

        # Act
        query = "test query"
        engine.search(query)

        # Assert - verify engine.search was called with correct formatted query
        expected_semantic_query = (
            "task: search result | query: test query"
            if use_prompt_templates
            else "test query"
        )

        mock_generate_embedding.assert_called_once_with(expected_semantic_query)
