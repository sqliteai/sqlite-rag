#!/usr/bin/env python3
"""Output formatters for CLI search results."""

from abc import ABC, abstractmethod
from typing import List, Optional

import typer

from sqlite_rag.models.sentence_result import SentenceResult

from .models.document_result import DocumentResult


class SearchResultFormatter(ABC):
    """Base class for search result formatters."""

    @abstractmethod
    def format_results(self, results: List[DocumentResult], query: str) -> None:
        """Format and display search results."""

    def _get_file_icon(self, uri: str) -> str:
        """Get appropriate icon for file type."""
        if not uri:
            return "📝"

        uri_lower = uri.lower()
        icon_map = {
            (".py", ".pyx"): "🐍",
            (".js", ".ts", ".jsx", ".tsx"): "⚡",
            (".md", ".markdown"): "📄",
            (".json", ".yaml", ".yml"): "📋",
            (".html", ".htm"): "🌐",
            (".css", ".scss", ".sass"): "🎨",
            (".txt", ".log"): "📃",
            (".pdf",): "📕",
            (".sql",): "🗃️",
        }

        for extensions, icon in icon_map.items():
            if any(uri_lower.endswith(ext) for ext in extensions):
                return icon
        return "📄"

    def _clean_and_wrap_snippet(
        self, snippet: str, width: int = 75, max_length: int = 400
    ) -> List[str]:
        """Clean snippet and wrap to specified width with max length limit."""
        # Clean the snippet
        clean = snippet.replace("\n", " ").replace("\r", "")
        clean = " ".join(clean.split())

        # Truncate to max length if needed
        if len(clean) > max_length:
            clean = clean[: max_length - 3] + "..."

        # Wrap to width
        lines = []
        words = clean.split()
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= width:
                current_line = current_line + " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def _format_uri_display(self, uri: str, icon: str, max_width: int = 75) -> str:
        """Format URI for display with icon and truncation."""
        if not uri:
            return ""

        uri_display = f"{icon} {uri}"
        if len(uri_display) > max_width:
            available_width = max_width - len(icon) - 4  # 4 for " ..."
            uri_display = f"{icon} ...{uri[-available_width:]}"
        return uri_display

    def _build_sentence_preview(
        self,
        chunk_content: str,
        sentences: List[SentenceResult],
        max_chars: int = 400,
    ) -> str:
        """Build preview from top 3 ranked sentences with [...] for gaps.

        Args:
            chunk_content: The full chunk text
            sentences: List of SentenceResult objects (should already be sorted by rank)
            max_chars: Maximum total characters for preview

        Returns:
            Preview string with top sentences and [...] separators.
            Falls back to truncated chunk_content if sentences have no offsets.
        """

        # Take top 3 sentences (they should already be sorted by rank/distance)
        top_sentences = sentences[:3] if sentences else []

        if not top_sentences:
            # Fallback: no sentences, return truncated chunk content
            return chunk_content[:max_chars]

        # Filter sentences that have offset information
        sentences_with_offsets = [
            s
            for s in top_sentences
            if s.start_offset is not None and s.end_offset is not None
        ]

        if not sentences_with_offsets:
            # Fallback: sentences exist but no offset information, return truncated chunk content
            return chunk_content[:max_chars]

        # Sort by start_offset to maintain document order
        sentences_with_offsets.sort(
            key=lambda s: s.start_offset if s.start_offset is not None else -1
        )

        preview_parts = []
        total_chars = 0
        prev_end_offset = None

        for sentence in sentences_with_offsets:
            # Extract sentence text using offsets
            sentence_text = chunk_content[
                sentence.start_offset : sentence.end_offset
            ].strip()

            # Calculate remaining budget including potential separator
            separator_len = len(" [...] ") if preview_parts else 0
            remaining = max_chars - total_chars - separator_len

            if remaining <= 0:
                break

            # Truncate sentence if needed
            if len(sentence_text) > remaining:
                sentence_text = sentence_text[: remaining - 3] + "..."

            # Check if there's a gap > 10 chars from previous sentence
            if prev_end_offset is not None and sentence.start_offset is not None:
                gap_size = sentence.start_offset - prev_end_offset
                if gap_size > 10:
                    preview_parts.append("[...]")
                    total_chars += len(" [...] ")

            preview_parts.append(sentence_text)
            total_chars += len(sentence_text)
            prev_end_offset = sentence.end_offset

        return " ".join(preview_parts)


class BoxedFormatter(SearchResultFormatter):
    """Base class for boxed result formatters."""

    def format_results(self, results: List[DocumentResult], query: str) -> None:
        if not results:
            typer.echo("No documents found matching the query.")
            return

        typer.echo(f"━━━ Search Results ({len(results)} matches) ━━━")
        typer.echo()

        for idx, doc in enumerate(results, 1):
            self._format_single_result(doc, idx)

    def _format_single_result(self, doc: DocumentResult, idx: int) -> None:
        """Format a single result with box layout."""
        icon = self._get_file_icon(doc.document.uri or "")

        # Use sentence-based preview if sentences are available
        if doc.sentences:
            snippet_text = self._build_sentence_preview(doc.snippet, doc.sentences)
        else:
            snippet_text = doc.snippet

        snippet_lines = self._clean_and_wrap_snippet(
            snippet_text, width=75, max_length=400
        )

        # Draw the result box header
        header = f"┌─ Result #{idx} " + "─" * (67 - len(str(idx)))
        typer.echo(header)

        # Display URI if available
        if doc.document.uri:
            uri_display = self._format_uri_display(doc.document.uri, icon, 75)
            typer.echo(f"│ {uri_display:<75}│")

            # Add debug info if needed
            debug_line = self._get_debug_line(doc)
            if debug_line:
                typer.echo(debug_line)

            typer.echo("├" + "─" * 77 + "┤")
        elif self._should_show_debug():
            debug_line = self._get_debug_line(doc)
            if debug_line:
                typer.echo(debug_line)
                typer.echo("├" + "─" * 77 + "┤")

        # Display snippet
        for line in snippet_lines:
            typer.echo(f"│ {line:<75} │")

        typer.echo("└" + "─" * 77 + "┘")
        typer.echo()

    def _get_debug_line(self, doc: DocumentResult) -> Optional[str]:
        """Get debug information line. Override in subclasses."""
        return None

    def _should_show_debug(self) -> bool:
        """Whether to show debug information. Override in subclasses."""
        return False


class BoxedDebugFormatter(BoxedFormatter):
    """Modern detailed formatter with debug information in boxes."""

    def _get_debug_line(self, doc: DocumentResult) -> str:
        """Format debug metrics line."""
        combined = (
            f"{doc.combined_rank:.5f}" if doc.combined_rank is not None else "N/A"
        )
        vec_info = (
            f"#{doc.vec_rank} ({doc.vec_distance:.6f})"
            if doc.vec_rank is not None
            else "N/A"
        )
        fts_info = (
            f"#{doc.fts_rank} ({doc.fts_score:.6f})"
            if doc.fts_rank is not None
            else "N/A"
        )
        return f"│ Combined: {combined} │ Vector: {vec_info} │ FTS: {fts_info}"

    def _should_show_debug(self) -> bool:
        return True

    def _format_single_result(self, doc: DocumentResult, idx: int) -> None:
        """Format a single result with box layout including sentence summary."""
        icon = self._get_file_icon(doc.document.uri or "")

        # Use sentence-based preview if sentences are available
        if doc.sentences:
            snippet_text = self._build_sentence_preview(doc.snippet, doc.sentences)
        else:
            snippet_text = doc.snippet

        snippet_lines = self._clean_and_wrap_snippet(
            snippet_text, width=75, max_length=400
        )

        # Draw the result box header
        header = f"┌─ Result #{idx} " + "─" * (67 - len(str(idx)))
        typer.echo(header)

        # Display URI if available
        if doc.document.uri:
            uri_display = self._format_uri_display(doc.document.uri, icon, 75)
            typer.echo(f"│ {uri_display:<75}│")

            # Add debug info
            debug_line = self._get_debug_line(doc)
            if debug_line:
                typer.echo(debug_line)

            typer.echo("├" + "─" * 77 + "┤")
        elif self._should_show_debug():
            debug_line = self._get_debug_line(doc)
            if debug_line:
                typer.echo(debug_line)
                typer.echo("├" + "─" * 77 + "┤")

        # Display snippet preview
        for line in snippet_lines:
            typer.echo(f"│ {line:<75} │")

        # Display sentence details if available
        if doc.sentences:
            typer.echo("├" + "─" * 77 + "┤")
            typer.echo(
                "│ Sentences:                                                                │"
            )

            for sentence in doc.sentences[:5]:  # Show max 5 sentences
                distance_str = (
                    f"{sentence.distance:.6f}"
                    if sentence.distance is not None
                    else "N/A"
                )
                rank_str = f"#{sentence.rank}" if sentence.rank is not None else "N/A"

                # Extract sentence preview (first 50 chars)
                if (
                    sentence.start_offset is not None
                    and sentence.end_offset is not None
                ):
                    sentence_text = doc.snippet[
                        sentence.start_offset : sentence.end_offset
                    ].strip()
                    # Truncate and clean for display
                    sentence_preview = sentence_text.replace("\n", " ").replace(
                        "\r", ""
                    )
                    if len(sentence_preview) > 50:
                        sentence_preview = sentence_preview[:47] + "..."
                else:
                    sentence_preview = "[No offset info]"

                # Format sentence line
                sentence_line = (
                    f"│   {rank_str:>3} ({distance_str}) | {sentence_preview}"
                )
                # Pad to 78 chars and add closing border
                typer.echo(sentence_line.ljust(78) + " │")

        typer.echo("└" + "─" * 77 + "┘")
        typer.echo()


class TableDebugFormatter(SearchResultFormatter):
    """Table view debug formatter."""

    def format_results(self, results: List[DocumentResult], query: str) -> None:
        if not results:
            typer.echo("No documents found matching the query.")
            return

        typer.echo(f"Found {len(results)} documents:")
        self._print_table_header()

        for idx, doc in enumerate(results, 1):
            self._print_table_row(idx, doc)

    def _print_table_header(self) -> None:
        """Print the table header."""
        headers = [
            "#",
            "Preview",
            "URI",
            "C.Rank",
            "V.Rank",
            "FTS.Rank",
            "V.Dist",
            "FTS.Score",
        ]
        widths = [3, 55, 35, 33, 8, 9, 18, 18]

        header_line = "".join(
            f"{header:<{width}}" for header, width in zip(headers, widths)
        )
        typer.echo(header_line)
        typer.echo("─" * sum(widths))

    def _print_table_row(self, idx: int, doc: DocumentResult) -> None:
        """Print a single table row."""
        # Use sentence-based preview if sentences are available
        if doc.sentences:
            snippet = self._build_sentence_preview(
                doc.snippet, doc.sentences, max_chars=52
            )
        else:
            snippet = doc.snippet

        # Clean snippet display
        snippet = snippet.replace("\n", " ").replace("\r", "")
        snippet = snippet[:49] + "..." if len(snippet) > 52 else snippet

        # Clean URI display
        uri = doc.document.uri or "N/A"
        uri = "..." + uri[-29:] if len(uri) > 32 else uri

        # Format debug values
        values = [
            str(idx),
            snippet,
            uri,
            f"{doc.combined_rank:.17f}" if doc.combined_rank is not None else "N/A",
            str(doc.vec_rank) if doc.vec_rank is not None else "N/A",
            str(doc.fts_rank) if doc.fts_rank is not None else "N/A",
            f"{doc.vec_distance:.6f}" if doc.vec_distance is not None else "N/A",
            f"{doc.fts_score:.6f}" if doc.fts_score is not None else "N/A",
        ]
        widths = [3, 55, 35, 33, 8, 9, 18, 18]

        row_line = "".join(f"{value:<{width}}" for value, width in zip(values, widths))
        typer.echo(row_line)


def get_formatter(
    debug: bool = False, table_view: bool = False
) -> SearchResultFormatter:
    """Factory function to get the appropriate formatter."""
    if table_view:
        return TableDebugFormatter()
    elif debug:
        return BoxedDebugFormatter()
    else:
        return BoxedFormatter()
