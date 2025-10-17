from sqlite_rag.models.chunk import Chunk
from sqlite_rag.sentence_splitter import SentenceSplitter


class TestSentenceSplitter:
    def test_split(self):

        splitter = SentenceSplitter()

        chunk = Chunk(
            id=1,
            document_id=1,
            title="Test Chunk",
            content="This is the first sentence.\nHere is the second sentence! And what about the third?",
            embedding=b"",
            sentences=[],
        )

        sentences = splitter.split(chunk)

        assert len(sentences) == 3
        assert sentences[0].content == "This is the first sentence."
        assert sentences[0].sequence == 0
        assert sentences[0].start_offset == 0
        assert sentences[0].end_offset == 27

        assert sentences[1].content == "Here is the second sentence!"
        assert sentences[1].sequence == 1
        assert sentences[1].start_offset == 28
        assert sentences[1].end_offset == 28 + 28

        assert sentences[2].content == "And what about the third?"
        assert sentences[2].sequence == 2
        assert sentences[2].start_offset == 57
        assert sentences[2].end_offset == 57 + 25

    def test_split_empty(self):
        splitter = SentenceSplitter()

        chunk = Chunk(
            id=1,
            document_id=1,
            title="Empty Chunk",
            content="",
            embedding=b"",
            sentences=[],
        )

        sentences = splitter.split(chunk)

        assert len(sentences) == 0

    def test_split_no_punctuation(self):
        splitter = SentenceSplitter()

        chunk = Chunk(
            id=1,
            document_id=1,
            title="No Punctuation Chunk",
            content="This is a sentence without punctuation and another one follows it",
            embedding=b"",
            sentences=[],
        )

        sentences = splitter.split(chunk)

        assert len(sentences) == 1
        assert sentences[0].content == chunk.content
        assert sentences[0].sequence == 0
        assert sentences[0].start_offset == 0
        assert sentences[0].end_offset == len(chunk.content)
