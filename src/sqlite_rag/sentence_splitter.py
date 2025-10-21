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
