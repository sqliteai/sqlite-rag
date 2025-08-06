import sqlite3
from typing import List

from .models.chunk import Chunk
from .settings import Settings
import math


class Chunker:
    def __init__(self, conn: sqlite3.Connection, settings: Settings):
        self._conn = conn
        self.settings = settings

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk text using Recursive Character Text Splitter."""
        if self._get_token_count(text) <= self.settings.chunk_size:
            return [Chunk(content=text)]

        return self._recursive_split(text)

    def _get_token_count(self, text: str) -> int:
        """Get token count using SQLite AI extension."""
        if text == "":
            return 0
        cursor = self._conn.execute("SELECT llm_token_count(?)", (text,))
        return cursor.fetchone()[0]

    def _estimate_tokens_count(self, text: str) -> int:
        """Estimate token count more conservatively."""
        # This is a simple heuristic; adjust as needed
        return (len(text) + 3) // 4

    def _recursive_split(self, text: str) -> List[Chunk]:
        """Recursively split text into chunks with overlap."""
        separators = [
            "\n\n",  # Double newlines (paragraphs)
            "\n",  # Single newlines
            " ",  # Spaces
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",  # Character level (fallback)
        ]

        return self._split_text_with_separators(text, separators)

    def _split_text_with_separators(
        self, text: str, separators: List[str]
    ) -> List[Chunk]:
        """Split text using hierarchical separators."""
        chunks = []

        if self.settings.chunk_size <= self.settings.chunk_overlap:
            raise ValueError("Chunk size must be greater than chunk overlap.")

        if not separators:
            # Fallback: character-level splitting
            return self._split_by_characters(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._split_by_characters(text)

        # Reserve space for overlap
        effective_chunk_size = max(
            1, self.settings.chunk_size - self.settings.chunk_overlap
        )

        splits = text.split(separator)
        current_chunk = ""

        for split in splits:
            test_chunk = current_chunk + (separator if current_chunk else "") + split

            if self._get_token_count(test_chunk) <= effective_chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(Chunk(content=current_chunk.strip()))

                # If single split is too large, recursively split it
                if self._get_token_count(split) > effective_chunk_size:
                    sub_chunks = self._split_text_with_separators(
                        split, remaining_separators
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # Add final chunk
        if current_chunk:
            chunks.append(Chunk(content=current_chunk.strip()))

        return self._apply_overlap(chunks)

    def _split_by_characters(self, text: str) -> List[Chunk]:
        """Split text at character level when no separators work."""
        chunks = []

        # Reserve space for overlap
        effective_chunk_size = max(
            1, self.settings.chunk_size - self.settings.chunk_overlap
        )

        total_tokens = self._get_token_count(text)
        chars_per_token = (
            math.ceil(len(text) / total_tokens)
            if total_tokens > 0
            else 4  # Assume 4 chars per token if no tokens found
        )

        # Estimate characters that fit the chunk size
        estimated_chunk_size_in_chars = int(effective_chunk_size * chars_per_token)

        start = 0
        while start < len(text):
            # The end position of the next chunk of text
            end = min(start + estimated_chunk_size_in_chars, len(text))

            chunk_text = text[start:end]

            # Verify it doesn't exceed token limit, reduce if needed
            while (
                self._get_token_count(chunk_text) > effective_chunk_size
                and end > start + 1
            ):
                end = int(end * 0.9)  # Reduce by 10%
                chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(Chunk(content=chunk_text.strip()))

            start = end

        return self._apply_overlap(chunks)

    def _apply_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.settings.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = [chunks[0]]  # First chunk has no overlap

        for i in range(1, len(chunks)):
            current_content = chunks[i].content
            prev_content = chunks[i - 1].content

            # Get overlap text from end of previous chunk
            overlap_text = self._get_overlap_text(
                prev_content, self.settings.chunk_overlap
            )

            if overlap_text:
                combined_content = overlap_text + " " + current_content
            else:
                combined_content = current_content

            overlapped_chunks.append(Chunk(content=combined_content))

        return overlapped_chunks

    def _get_overlap_text(self, text: str, max_overlap_tokens: int) -> str:
        """Extract overlap text from end of text, respecting token limit."""
        words = text.split()
        if not words:
            return ""

        # Try to find the longest suffix that fits within overlap limit
        for i in range(len(words)):
            suffix = " ".join(words[i:])
            if self._get_token_count(suffix) <= max_overlap_tokens:
                return suffix

        # If even single word is too large, return empty
        return ""
