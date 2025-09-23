from pathlib import Path

import pytest

from sqlite_rag.chunker import Chunker
from sqlite_rag.settings import Settings


class MockCursor:
    """Mock cursor that returns token count."""

    def __init__(self, token_count):
        self.token_count = token_count

    def fetchone(self):
        return {"count": self.token_count}


class MockSQLiteConnection:
    """Mock SQLite connection that simulates token counting."""

    def __init__(self):
        self.call_count = 0

    def execute(self, query, params=None):
        """Mock execute method that returns token count based on text length."""
        self.call_count += 1
        if query.startswith("SELECT llm_token_count"):
            text = params[0] if params else ""
            # Simple token estimation: ~4 characters per token
            token_count = max(1, len(text) // 4)
            return MockCursor(token_count)
        return MockCursor(0)


@pytest.fixture
def mock_conn():
    """Fixture providing a mock SQLite connection."""
    return MockSQLiteConnection()


@pytest.fixture
def chunker_large(mock_conn):
    """Fixture providing a chunker with large chunk size."""
    settings = Settings("test-model", use_prompt_templates=False)
    settings.chunk_size = 100
    settings.chunk_overlap = 20
    return Chunker(mock_conn, settings)


@pytest.fixture
def chunker_small(mock_conn):
    """Fixture providing a chunker with small chunk size."""
    settings = Settings("test-model", use_prompt_templates=False)
    settings.chunk_size = 25
    settings.chunk_overlap = 5
    return Chunker(mock_conn, settings)


@pytest.fixture
def chunker_tiny(mock_conn):
    """Fixture providing a chunker with tiny chunk size."""
    settings = Settings("test-model", use_prompt_templates=False)
    settings.chunk_size = 8
    settings.chunk_overlap = 2
    return Chunker(mock_conn, settings)


class TestSingleChunk:
    """Test cases for single chunk scenarios."""

    def test_short_text_single_chunk(self, chunker_large):
        """Test that short text returns a single chunk."""
        text = "This is a short text that should fit in a single chunk."
        chunks = chunker_large.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_empty_text(self, chunker_large):
        """Test empty text handling."""
        text = ""
        chunks = chunker_large.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == ""

    def test_chunk_enrichness_with_input_title(self, chunker_large):
        """Test that chunk enrichment adds metadata correctly."""
        text = "This is a test chunk."
        metadata = {"title": "Test Title"}

        chunks = chunker_large.chunk(text, metadata)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].title == "Test Title"

    def test_chunk_enrichness_with_generated_title(self, chunker_large):
        text = "# My title\n\nThis is a paragraph to test chunk."
        metadata = {"generated": {"title": "My title"}}

        chunks = chunker_large.chunk(text, metadata)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].title == "My title"


class TestParagraphSplitting:
    """Test cases for paragraph-level splitting."""

    def test_multiple_paragraphs(self, chunker_small):
        """Test splitting by paragraphs."""
        text = """First paragraph with some content here.

Second paragraph with different content.

Third paragraph with more text."""

        chunks = chunker_small.chunk(text)

        assert len(chunks) > 1
        # Verify we have meaningful content in chunks
        assert all(chunk.content.strip() for chunk in chunks)


class TestSentenceSplitting:
    """Test cases for sentence-level splitting."""

    def test_sentence_boundaries(self, chunker_tiny):
        """Test splitting respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker_tiny.chunk(text)

        assert len(chunks) > 1
        # Most chunks should contain periods (sentence endings)
        period_chunks = sum(1 for chunk in chunks if "." in chunk.content)
        assert period_chunks >= 1


class TestWordSplitting:
    """Test cases for word-level splitting."""

    def test_word_boundaries(self, chunker_tiny):
        """Test splitting by spaces (words)."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"

        chunks = chunker_tiny.chunk(text)

        assert len(chunks) > 1
        # Each chunk should contain complete words
        for chunk in chunks:
            if chunk.content.strip():
                words = chunk.content.split()
                assert len(words) >= 1


class TestCharacterFallback:
    """Test cases for character-level fallback splitting."""

    def test_no_separators_fallback(self, chunker_tiny):
        """Test fallback to character splitting when no separators work."""
        # Very long word with no separators
        text = "supercalifragilisticexpialidociousthisisaverylongwordwithoutanyspacesorpunctuationatall"

        chunks = chunker_tiny.chunk(text)

        assert len(chunks) > 1
        # Should create multiple chunks from the long word
        assert all(chunk.content.strip() for chunk in chunks)

    def test_character_splitting_token_limit(self, chunker_tiny):
        """Test that character splitting respects token limits."""
        text = "verylongwordwithoutanybreakpointsandshouldbesplitbycharacters"

        chunks = chunker_tiny.chunk(text)

        # Each chunk should respect the chunk size limit
        for chunk in chunks:
            token_count = chunker_tiny._get_token_count(chunk.content)
            assert token_count <= chunker_tiny._settings.chunk_size


class TestOverlapFunctionality:
    """Test cases for overlap functionality."""

    def test_overlap_between_chunks(self, chunker_small):
        """Test that overlap exists between consecutive chunks."""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."

        chunks = chunker_small.chunk(text)

        if len(chunks) > 1:
            # With overlap, chunks should share some content
            for i in range(1, len(chunks)):
                prev_words = set(chunks[i - 1].content.lower().split())
                curr_words = set(chunks[i].content.lower().split())

                # Should have some word overlap
                common_words = prev_words & curr_words
                assert len(common_words) > 0, f"No overlap between chunks {i-1} and {i}"

    def test_no_overlap_setting(self, mock_conn):
        """Test chunking without overlap."""
        settings = Settings("test-model")
        settings.chunk_size = 20
        settings.chunk_overlap = 0  # No overlap

        chunker = Chunker(mock_conn, settings)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_overlap_applied_only_once(self, mock_conn):
        """Test that overlap is applied only once, even when text goes through multiple separator levels."""
        settings = Settings("test-model")
        settings.chunk_size = 30  # Small chunk size to force splitting
        settings.chunk_overlap = 8  # Significant overlap

        chunker = Chunker(mock_conn, settings)

        # Create text that will be split by multiple separators:
        # 1. First by paragraphs (\n\n)
        # 2. Then by sentences (.)
        # 3. Finally by words ( )
        text = "This is the first paragraph with multiple sentences. This should be split across separators.\n\nThis is the second paragraph with more content. This will also be split by multiple separators and should trigger the overlap bug."

        chunks = chunker.chunk(text)

        # Verify that no chunk exceeds the chunk_size limit
        # If overlap is applied multiple times, chunks will be longer than chunk_size
        for i, chunk in enumerate(chunks):
            token_count = chunker._get_token_count(chunk.content)
            assert token_count <= settings.chunk_size, (
                f"Chunk {i} exceeds size limit: {token_count} tokens > {settings.chunk_size} tokens. "
                f"Content: '{chunk.content[:100]}...'"
            )

    def test_chunk_size_equals_overlap(self, mock_conn):
        """Test when chunk_size equals chunk_overlap."""
        settings = Settings("test-model")
        settings.chunk_size = 10
        settings.chunk_overlap = 10

        chunker = Chunker(mock_conn, settings)
        text = "This is a test sentence that should be handled gracefully."

        with pytest.raises(ValueError) as excinfo:
            chunker.chunk(text)
        assert "Chunk size must be greater than chunk overlap." in str(excinfo.value)

    def test_very_small_chunk_size(self, mock_conn):
        """Test with chunk_size = 1."""
        settings = Settings("test-model")
        settings.chunk_size = 1
        settings.chunk_overlap = 0

        chunker = Chunker(mock_conn, settings)
        text = "Short text."

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_split_by_character_with_long_string(self, chunker_large):
        """Long string to be split by characters should be split in chunks
        with similar size, not become zone-length and loop forever."""
        with open(Path(__file__).parent / "assets" / "doc-base64-images.md", "r") as f:
            text = f.read()

        chunks = chunker_large.chunk(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content)
            assert (
                chunker_large._get_token_count(chunk.content)
                <= chunker_large._settings.chunk_size
            ), pytest.fail(
                f"Chunk exceeds size limit: {chunker_large._get_token_count(chunk.content)} tokens > {chunker_large._settings.chunk_size} tokens"
            )
