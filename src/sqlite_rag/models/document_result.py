from dataclasses import dataclass, field

from .document import Document
from .sentence_result import SentenceResult


@dataclass
class DocumentResult:
    document: Document

    chunk_id: int
    chunk_content: str

    combined_rank: float
    vec_rank: float | None = None
    fts_rank: float | None = None

    vec_distance: float | None = None
    fts_score: float | None = None

    # highlight sentences
    sentences: list[SentenceResult] = field(default_factory=list)

    def get_preview(
        self, top_k_sentences: int = 3, max_chars: int = 400, gap: str = "[...]"
    ) -> str:
        """Build preview from top ranked sentences with [...] for gaps.

        Args:
            top_k_sentences: Number of top sentences to include in preview
            max_chars: Maximum total characters for preview

        Returns:
            Preview string with top sentences and [...] separators.
            Falls back to truncated chunk_content if sentences have no offsets.
        """
        top_sentences = self.sentences[:top_k_sentences] if self.sentences else []

        if not top_sentences:
            # Fallback: no sentences, return truncated chunk content
            return self.chunk_content[:max_chars]

        # Filter sentences that have offset information
        sentences_with_offsets = [
            s
            for s in top_sentences
            if s.start_offset is not None and s.end_offset is not None
        ]

        if not sentences_with_offsets:
            return self.chunk_content[:max_chars]

        # Sort by start_offset to maintain document order
        sentences_with_offsets.sort(
            key=lambda s: s.start_offset if s.start_offset is not None else -1
        )

        preview_parts = []
        total_chars = 0
        prev_end_offset = None

        for sentence in sentences_with_offsets:
            sentence_text = self.chunk_content[
                sentence.start_offset : sentence.end_offset
            ].strip()

            # Calculate remaining budget including potential separator
            separator_len = len("[...] ") if preview_parts else 0
            remaining = max_chars - total_chars - separator_len

            if remaining <= 0:
                break

            if prev_end_offset is not None and sentence.start_offset is not None:
                gap_size = sentence.start_offset - prev_end_offset
                if gap_size > 10:
                    preview_parts.append(gap)
                    total_chars += len(gap)

            preview_parts.append(sentence_text)
            total_chars += len(sentence_text)
            prev_end_offset = sentence.end_offset

        preview = " ".join(preview_parts)

        return preview[: max_chars - 3] + "..." if len(preview) > max_chars else preview
