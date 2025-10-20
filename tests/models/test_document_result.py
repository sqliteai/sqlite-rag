from sqlite_rag.models.document import Document
from sqlite_rag.models.document_result import DocumentResult
from sqlite_rag.models.sentence_result import SentenceResult


class TestDocumentResult:
    def test_get_preview_no_sentences(self):
        doc = Document(uri="test.txt", content="test content")
        chunk_content = "This is a long piece of text. " * 50
        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content=chunk_content,
            combined_rank=1.0,
            sentences=[],
        )

        preview = result.get_preview(max_chars=100)
        assert len(preview) == 100
        assert preview == chunk_content[:100]

    def test_get_preview_with_sentences_no_offsets(self):
        doc = Document(uri="test.txt", content="test content")
        chunk_content = "First sentence. Second sentence. Third sentence."

        # Sentences without offset information
        sentences = [
            SentenceResult(chunk_id=1, id=1, rank=1, distance=0.1),
            SentenceResult(chunk_id=1, id=2, rank=2, distance=0.2),
        ]

        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content=chunk_content,
            combined_rank=1.0,
            sentences=sentences,
        )

        preview = result.get_preview(max_chars=100)
        assert preview == chunk_content[:100]

    def test_get_preview_with_single_sentence(self):
        doc = Document(uri="test.txt", content="test content")
        chunk_content = (
            "First sentence here. Second sentence there. Third sentence everywhere."
        )

        sentences = [
            SentenceResult(
                chunk_id=1,
                id=2,
                rank=1,
                distance=0.1,
                start_offset=21,
                end_offset=44,
            ),
        ]

        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content=chunk_content,
            combined_rank=1.0,
            sentences=sentences,
        )

        preview = result.get_preview(max_chars=400)
        assert preview == "Second sentence there."

    def test_get_preview_with_gaps(self):
        """Test get_preview adds [...] separator for gaps."""
        doc = Document(uri="test.txt", content="test content")
        chunk_content = (
            "First sentence at the beginning."
            "Some middle content that we skip over here."
            "Last sentence at the end."
        )

        sentences = [
            SentenceResult(
                chunk_id=1,
                id=1,
                rank=1,
                distance=0.1,
                start_offset=0,
                end_offset=32,  # "First sentence at the beginning."
            ),
            SentenceResult(
                chunk_id=1,
                id=3,
                rank=2,
                distance=0.2,
                start_offset=75,
                end_offset=103,  # "Last sentence at the end."
            ),
        ]

        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content=chunk_content,
            combined_rank=1.0,
            sentences=sentences,
        )

        preview = result.get_preview(max_chars=400)
        assert (
            "First sentence at the beginning. [...] Last sentence at the end."
            == preview
        )

    def test_get_preview_respects_max_chars(self):
        """Test get_preview truncates when exceeding max_chars."""
        doc = Document(uri="test.txt", content="test content")
        chunk_content = (
            "A very long sentence that exceeds the maximum character limit. " * 10
        )

        sentences = [
            SentenceResult(
                chunk_id=1,
                id=1,
                rank=1,
                distance=0.1,
                start_offset=0,
                end_offset=200,
            ),
        ]

        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content=chunk_content,
            combined_rank=1.0,
            sentences=sentences,
        )

        preview = result.get_preview(max_chars=50)
        assert len(preview) <= 50

    def test_get_preview_with_multiple_consecutive_and_ordered_sentences(self):
        doc = Document(uri="test.txt", content="test content")
        chunk_content = "First sentence. Second sentence. Third sentence."

        sentences = [
            SentenceResult(
                chunk_id=1,
                id=1,
                rank=1,
                distance=0.1,
                start_offset=0,
                end_offset=15,
            ),
            SentenceResult(
                chunk_id=1,
                id=2,
                rank=2,
                distance=0.2,
                start_offset=16,
                end_offset=32,
            ),
        ]

        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content=chunk_content,
            combined_rank=1.0,
            sentences=sentences,
        )

        preview = result.get_preview(max_chars=400)
        assert preview == "First sentence. Second sentence."

    def test_get_preview_orders_sentences_by_offset(self):
        """Test get_preview reorders sentences by start_offset (document order)."""
        doc = Document(uri="test.txt", content="test content")
        chunk_content = "First sentence. " + "x" * 50 + " Third sentence."

        # Sentences in reverse rank order (rank 1 is last in document)
        sentences = [
            SentenceResult(
                chunk_id=1,
                id=3,
                rank=1,  # higher rank but appears latter in document
                distance=0.1,
                start_offset=66,
                end_offset=82,  # "Third sentence."
            ),
            SentenceResult(
                chunk_id=1,
                id=1,
                rank=2,
                distance=0.2,
                start_offset=0,
                end_offset=15,  # "First sentence."
            ),
        ]

        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content=chunk_content,
            combined_rank=1.0,
            sentences=sentences,
        )

        preview = result.get_preview(max_chars=400)
        # Should be in document order despite rank order
        assert "First sentence. [...] Third sentence." == preview

    def test_get_preview_limits_to_top_k_sentences(self):
        """Test get_preview respects top_k_sentences parameter."""
        doc = Document(uri="test.txt", content="test content")
        chunk_content = "First. Second. Third. Fourth. Fifth."

        # 5 sentences, but only top 2 should be used
        sentences = [
            SentenceResult(
                chunk_id=1, id=1, rank=1, distance=0.1, start_offset=0, end_offset=6
            ),
            SentenceResult(
                chunk_id=1, id=2, rank=2, distance=0.2, start_offset=7, end_offset=14
            ),
            SentenceResult(
                chunk_id=1, id=3, rank=3, distance=0.3, start_offset=15, end_offset=21
            ),
            SentenceResult(
                chunk_id=1, id=4, rank=4, distance=0.4, start_offset=22, end_offset=29
            ),
            SentenceResult(
                chunk_id=1, id=5, rank=5, distance=0.5, start_offset=30, end_offset=36
            ),
        ]

        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content=chunk_content,
            combined_rank=1.0,
            sentences=sentences,
        )

        preview = result.get_preview(top_k_sentences=2, max_chars=400)
        assert "First." in preview
        assert "Second." in preview
        assert "Third" not in preview
        assert "Fourth" not in preview
        assert "Fifth" not in preview

        # Test with default top_k=3
        preview_default = result.get_preview(max_chars=400)
        assert "First." in preview_default
        assert "Second." in preview_default
        assert "Third." in preview_default
        assert "Fourth" not in preview_default
        assert "Fifth" not in preview_default
