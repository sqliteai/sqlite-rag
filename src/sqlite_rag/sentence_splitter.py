import re
from typing import List

from sqlite_rag.models.chunk import Chunk
from sqlite_rag.models.sentence import Sentence


class SentenceSplitter:
    MIN_CHARS_PER_SENTENCE = 20

    def split(self, chunk: Chunk) -> List[Sentence]:
        """Split chunk into sentences."""
        sentences = []

        sentences_text = self._split_into_sentences(chunk.content)
        start_offset = 0
        end_offset = 0
        for sentence_text in sentences_text:
            start_offset = chunk.content.index(sentence_text, end_offset)
            end_offset = start_offset + len(sentence_text)

            sentence = Sentence(
                content=sentence_text,
                start_offset=start_offset,
                end_offset=end_offset,
            )
            sentences.append(sentence)

        return sentences

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split into focused segments for semantic matching."""
        # Split on: sentence endings, semicolons, or paragraph breaks
        sentence_endings = re.compile(r'(?<=[.!?;])(?:"|\')?\s+(?=[A-Z])|[\n]{2,}')
        sentences = sentence_endings.split(text)

        # Keep segments that are substantial enough (20+ chars for meaningful matching)
        return [
            s.strip() for s in sentences if len(s.strip()) > self.MIN_CHARS_PER_SENTENCE
        ]
