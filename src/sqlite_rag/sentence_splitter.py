import re
from typing import List

from sqlite_rag.models.chunk import Chunk
from sqlite_rag.models.sentence import Sentence


class SentenceSplitter:
    MIN_CHARS_PER_SENTENCE = 20

    def split(self, chunk: Chunk) -> List[Sentence]:
        """Split chunk into sentences."""
        # Split on: sentence endings, semicolons, or paragraph breaks
        sentence_regex = re.compile(r'(?<=[.!?;])(?:"|\')?\s+(?=[A-Z])|[\n]{2,}')

        sentences = []
        last_end = 0
        text = chunk.content

        for match in sentence_regex.finditer(text):
            segment = text[last_end : match.end()]

            segment = segment.strip()
            if len(segment) > self.MIN_CHARS_PER_SENTENCE:
                sentences.append(
                    Sentence(
                        content=segment,
                        start_offset=last_end,
                        end_offset=last_end + len(segment),
                    )
                )

            # Position after the current match
            last_end = match.end()

        # Last segment
        if last_end < len(text):
            segment = text[last_end:]

            segment = segment.strip()
            if len(segment) > self.MIN_CHARS_PER_SENTENCE:
                sentences.append(
                    Sentence(
                        content=segment,
                        start_offset=last_end,
                        end_offset=last_end + len(segment),
                    )
                )

        return sentences

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split into focused segments for semantic matching."""

        sentence_endings = re.compile(r'(?<=[.!?;])(?:"|\')?\s+(?=[A-Z])|[\n]{2,}')
        sentences = sentence_endings.split(text)

        # Keep segments that are substantial enough (20+ chars for meaningful matching)
        return [
            s.strip() for s in sentences if len(s.strip()) > self.MIN_CHARS_PER_SENTENCE
        ]
