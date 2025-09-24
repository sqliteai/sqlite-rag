import math
import sqlite3
from typing import List, Optional

from sqlite_rag.models.document import Document

from .models.chunk import Chunk
from .settings import Settings


class Chunker:
    ESTIMATE_CHARS_PER_TOKEN = 4

    def __init__(self, conn: sqlite3.Connection, settings: Settings):
        self._conn = conn
        self._settings = settings

    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk text using Recursive Character Text Splitter."""
        chunk = self._create_chunk(document.content, title=document.get_title())

        if (
            self._get_token_count(chunk.get_embedding_text())
            <= self._settings.chunk_size
        ):
            return [chunk]

        return self._recursive_split(document)

    def _create_chunk(
        self,
        content: str,
        head_overlap_text: str = "",
        title: Optional[str] = None,
    ) -> Chunk:
        prompt = None
        if self._settings.use_prompt_templates:
            prompt = self._settings.prompt_template_retrieval_document

        return Chunk(
            content=content,
            head_overlap_text=head_overlap_text,
            prompt=prompt,
            title=title,
        )

    def _get_effective_chunk_size(self, prompt: str) -> int:
        """Calculate effective chunk size considering overlap and other
        prompt data useful to the model.

        Args:
            prompt: The prompt template without content.
        """
        if self._settings.chunk_size <= self._settings.chunk_overlap:
            raise ValueError("Chunk size must be greater than chunk overlap.")

        prompt_size = self._get_token_count(prompt)
        return self._settings.chunk_size - self._settings.chunk_overlap - prompt_size

    def _get_token_count(self, text: str) -> int:
        """Get token count using SQLite AI extension."""
        if text == "":
            return 0

        # Fallback to estimated token count for very large texts
        # to avoid performance issues
        if len(text) > self._settings.chunk_size * self.ESTIMATE_CHARS_PER_TOKEN * 2:
            return self._estimate_tokens_count(text)

        cursor = self._conn.execute("SELECT llm_token_count(?) AS count", (text,))
        return cursor.fetchone()["count"]

    def _estimate_tokens_count(self, text: str) -> int:
        """Estimate token count more conservatively."""
        # This is a simple heuristic; adjust as needed
        return (len(text) + 3) // self.ESTIMATE_CHARS_PER_TOKEN

    def _recursive_split(self, document: Document) -> List[Chunk]:
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

        empty_chunk = self._create_chunk("", title=document.get_title())
        effective_chunk_size = max(
            1, self._get_effective_chunk_size(empty_chunk.get_embedding_text())
        )

        chunks_content = self._split_text_with_separators(
            document.content, separators, effective_chunk_size
        )
        overlaps = self._create_overlaps(chunks_content)

        assert len(chunks_content) == len(overlaps), "Mismatch in chunks and overlaps"
        return [
            self._create_chunk(
                content=chunk, head_overlap_text=overlap, title=document.get_title()
            )
            for chunk, overlap in zip(chunks_content, overlaps)
        ]

    def _split_text_with_separators(
        self, text: str, separators: List[str], effective_chunk_size: int
    ) -> List[str]:
        """Split text using hierarchical separators.
        Args:
            text: The text to split.
            separators: List of separators to use in order.
            effective_chunk_size: Reserved space for actual chunk content.
        """
        chunks = []

        if self._settings.chunk_size <= self._settings.chunk_overlap:
            raise ValueError("Chunk size must be greater than chunk overlap.")

        if not separators:
            # Fallback: character-level splitting
            return self._split_by_characters(text, effective_chunk_size)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._split_by_characters(text, effective_chunk_size)

        splits = text.split(separator)
        current_chunk = ""

        for split in splits:
            test_chunk = current_chunk + (separator if current_chunk else "") + split

            if self._get_token_count(test_chunk) <= effective_chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)

                # If single split is too large, recursively split it
                if self._get_token_count(split) > effective_chunk_size:
                    sub_chunks = self._split_text_with_separators(
                        split, remaining_separators, effective_chunk_size
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_characters(self, text: str, effective_chunk_size: int) -> List[str]:
        """Split text at character level when no separators work."""
        chunks = []

        total_tokens = self._get_token_count(text)
        chars_per_token = (
            math.ceil(len(text) / total_tokens)
            if total_tokens > 0
            else self.ESTIMATE_CHARS_PER_TOKEN  # Assume chars per token if no tokens found
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
                attempt_chunk_size = int((end - start) * 0.9)  # Reduce by 10%
                end = start + attempt_chunk_size
                chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(chunk_text)

            start = end

        return chunks

    def _create_overlaps(self, chunks: List[str]) -> List[str]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1 or self._settings.chunk_overlap <= 0:
            # Empty overlap for each chunk
            return [""] * len(chunks)

        overlapped_chunks = [""]  # First chunk has no overlap

        for i in range(1, len(chunks)):
            prev_content = chunks[i - 1]

            # Get overlap text from end of previous chunk
            overlap_text = self._get_overlap_text(
                prev_content, self._settings.chunk_overlap
            )

            overlapped_chunks.append(overlap_text)

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
