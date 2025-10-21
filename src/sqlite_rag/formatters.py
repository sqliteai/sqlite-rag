#!/usr/bin/env python3
"""Output formatters for CLI search results."""

from abc import ABC, abstractmethod
from typing import List

import typer

from .models.document_result import DocumentResult

# Display constants
BOX_CONTENT_WIDTH = 75
BOX_TOTAL_WIDTH = 77
SNIPPET_MAX_LENGTH = 400
SENTENCE_PREVIEW_LENGTH = 50
MAX_SENTENCES_DISPLAY = 5


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
        self,
        snippet: str,
        width: int = BOX_CONTENT_WIDTH,
        max_length: int = SNIPPET_MAX_LENGTH,
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

    def _format_uri_display(
        self, uri: str, icon: str, max_width: int = BOX_CONTENT_WIDTH
    ) -> str:
        """Format URI for display with icon and truncation."""
        if not uri:
            return ""

        uri_display = f"{icon} {uri}"
        if len(uri_display) > max_width:
            available_width = max_width - len(icon) - 4  # 4 for " ..."
            uri_display = f"{icon} ...{uri[-available_width:]}"
        return uri_display


class BoxedFormatter(SearchResultFormatter):
    """Boxed formatter for search results with optional debug information."""

    def __init__(self, show_debug: bool = False):
        """Initialize formatter.

        Args:
            show_debug: Whether to show debug information and sentence details
        """
        self.show_debug = show_debug

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
        snippet_text = doc.get_preview(max_chars=SNIPPET_MAX_LENGTH)
        snippet_lines = self._clean_and_wrap_snippet(snippet_text)

        # Draw box header
        header = f"┌─ Result #{idx} " + "─" * (BOX_TOTAL_WIDTH - 10 - len(str(idx)))
        typer.echo(header)

        # Display URI and debug info
        if doc.document.uri:
            uri_display = self._format_uri_display(doc.document.uri, icon)
            typer.echo(f"│ {uri_display:<{BOX_CONTENT_WIDTH}}│")

            if self.show_debug:
                self._print_debug_line(doc)

            typer.echo("├" + "─" * BOX_TOTAL_WIDTH + "┤")
        elif self.show_debug:
            self._print_debug_line(doc)
            typer.echo("├" + "─" * BOX_TOTAL_WIDTH + "┤")

        # Display snippet
        for line in snippet_lines:
            typer.echo(f"│ {line:<{BOX_CONTENT_WIDTH}} │")

        # Display sentence details in debug mode
        if self.show_debug and doc.sentences:
            self._print_sentence_details(doc)

        typer.echo("└" + "─" * BOX_TOTAL_WIDTH + "┘")
        typer.echo()

    def _print_debug_line(self, doc: DocumentResult) -> None:
        """Print debug metrics line."""
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
        debug_line = f"│ Combined: {combined} │ Vector: {vec_info} │ FTS: {fts_info}"
        typer.echo(debug_line)

    def _print_sentence_details(self, doc: DocumentResult) -> None:
        """Print sentence-level details."""
        typer.echo("├" + "─" * BOX_TOTAL_WIDTH + "┤")
        typer.echo(f"│ Sentences:{' ' * (BOX_CONTENT_WIDTH - 10)}│")

        for sentence in doc.sentences[:MAX_SENTENCES_DISPLAY]:
            distance_str = (
                f"{sentence.distance:.6f}" if sentence.distance is not None else "N/A"
            )
            rank_str = f"#{sentence.rank}" if sentence.rank is not None else "N/A"

            # Extract sentence preview
            if sentence.start_offset is not None and sentence.end_offset is not None:
                sentence_text = doc.chunk_content[
                    sentence.start_offset : sentence.end_offset
                ].strip()
                sentence_preview = sentence_text.replace("\n", " ").replace("\r", "")
                if len(sentence_preview) > SENTENCE_PREVIEW_LENGTH:
                    sentence_preview = (
                        sentence_preview[: SENTENCE_PREVIEW_LENGTH - 3] + "..."
                    )
            else:
                sentence_preview = "[No offset info]"

            # Format and print sentence line
            sentence_line = f"│   {rank_str:>3} ({distance_str}) | {sentence_preview}"
            typer.echo(sentence_line.ljust(BOX_TOTAL_WIDTH + 1) + " │")


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
        # Get snippet from DocumentResult (handles sentence-based preview automatically)
        snippet = doc.get_preview(max_chars=52)

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
    """Factory function to get the appropriate formatter.

    Args:
        debug: Show debug information and sentence details
        table_view: Use table format instead of boxed format

    Returns:
        SearchResultFormatter instance
    """
    if table_view:
        return TableDebugFormatter()
    return BoxedFormatter(show_debug=debug)
