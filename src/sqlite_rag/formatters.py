#!/usr/bin/env python3
"""Output formatters for CLI search results."""

from abc import ABC, abstractmethod
from typing import List

import typer

from .models.document_result import DocumentResult


class SearchResultFormatter(ABC):
    """Base class for search result formatters."""

    @abstractmethod
    def format_results(self, results: List[DocumentResult], query: str) -> None:
        """Format and display search results."""


class ModernCompactFormatter(SearchResultFormatter):
    """Modern compact formatter with better space utilization."""

    def format_results(self, results: List[DocumentResult], query: str) -> None:
        if not results:
            typer.echo("No documents found matching the query.")
            return

        typer.echo(f"━━━ Search Results ({len(results)} matches) ━━━")
        typer.echo()

        for idx, doc in enumerate(results, 1):
            # Get file type icon
            icon = self._get_file_icon(doc.document.uri or "")

            # Format title (use filename or "Text Content")
            self._get_document_title(doc.document.uri or "")

            # Clean and format content snippet
            snippet = self._clean_snippet(doc.snippet, max_length=200)

            # Display the result
            typer.echo(f"[{idx}]")
            if doc.document.uri:
                typer.echo(f"    {icon} {doc.document.uri}")
            typer.echo(f"    {snippet}")
            typer.echo("    " + "─" * 80)

    def _get_file_icon(self, uri: str) -> str:
        """Get appropriate icon for file type."""
        if not uri:
            return "📝"

        uri_lower = uri.lower()
        if uri_lower.endswith((".py", ".pyx")):
            return "🐍"
        elif uri_lower.endswith((".js", ".ts", ".jsx", ".tsx")):
            return "⚡"
        elif uri_lower.endswith((".md", ".markdown")):
            return "📄"
        elif uri_lower.endswith((".json", ".yaml", ".yml")):
            return "📋"
        elif uri_lower.endswith((".html", ".htm")):
            return "🌐"
        elif uri_lower.endswith((".css", ".scss", ".sass")):
            return "🎨"
        elif uri_lower.endswith((".txt", ".log")):
            return "📃"
        elif uri_lower.endswith((".pdf",)):
            return "📕"
        elif uri_lower.endswith((".sql",)):
            return "🗃️"
        else:
            return "📄"

    def _get_document_title(self, uri: str) -> str:
        """Extract a readable title from the document URI."""
        if not uri:
            return "Text Content"

        # Extract filename from path
        filename = uri.split("/")[-1] if "/" in uri else uri
        # Remove extension for cleaner display
        if "." in filename:
            return (
                filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
            )
        return filename.replace("_", " ").replace("-", " ").title()

    def _clean_snippet(self, snippet: str, max_length: int = 200) -> str:
        """Clean and truncate snippet for display."""
        # Replace newlines and multiple spaces
        clean = snippet.replace("\n", " ").replace("\r", "")
        # Collapse multiple spaces
        clean = " ".join(clean.split())

        if len(clean) > max_length:
            clean = clean[: max_length - 3] + "..."

        return clean


class ModernDetailedFormatter(SearchResultFormatter):
    """Modern detailed formatter with debug information in boxes."""

    def format_results(self, results: List[DocumentResult], query: str) -> None:
        if not results:
            typer.echo("No documents found matching the query.")
            return

        typer.echo(f"━━━ Search Results ({len(results)} matches) ━━━")
        typer.echo()

        for idx, doc in enumerate(results, 1):
            # Get file type icon and title
            icon = ModernCompactFormatter()._get_file_icon(doc.document.uri or "")
            ModernCompactFormatter()._get_document_title(doc.document.uri or "")

            # Format metrics
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

            # Clean snippet
            snippet = self._clean_and_wrap_snippet(doc.snippet, width=75)

            # Draw the result box
            typer.echo(f"┌─ Result #{idx} " + "─" * (67 - len(str(idx))))
            if doc.document.uri:
                uri_display = f"{icon} {doc.document.uri}"
                if len(uri_display) > 75:
                    uri_display = f"{icon} ...{doc.document.uri[-70:]}"
                typer.echo(f"│ {uri_display:<75}│")
            typer.echo(f"│ Combined: {combined} │ Vector: {vec_info} │ FTS: {fts_info}")
            typer.echo("├" + "─" * 77 + "┤")

            # Display snippet with proper wrapping
            for line in snippet:
                typer.echo(f"│ {line:<75} │")

            typer.echo("└" + "─" * 77 + "┘")
            typer.echo()

    def _clean_and_wrap_snippet(self, snippet: str, width: int = 75) -> List[str]:
        """Clean snippet and wrap to specified width."""
        # Clean the snippet
        clean = snippet.replace("\n", " ").replace("\r", "")
        clean = " ".join(clean.split())

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

        # Limit to reasonable number of lines
        if len(lines) > 4:
            lines = lines[:3] + [lines[3][: width - 3] + "..."]

        return lines


class TableDebugFormatter(SearchResultFormatter):
    """Legacy debug formatter for backwards compatibility."""

    def format_results(self, results: List[DocumentResult], query: str) -> None:
        if not results:
            typer.echo("No documents found matching the query.")
            return

        typer.echo(f"Found {len(results)} documents:")

        # Enhanced debug table with better formatting
        typer.echo(
            f"{'#':<3} {'Preview':<55} {'URI':<35} {'C.Rank':<33} {'V.Rank':<8} {'FTS.Rank':<9} {'V.Dist':<18} {'FTS.Score':<18}"
        )
        typer.echo("─" * 180)

        for idx, doc in enumerate(results, 1):
            # Clean snippet display
            snippet = doc.snippet.replace("\n", " ").replace("\r", "")
            if len(snippet) > 52:
                snippet = snippet[:49] + "..."

            # Clean URI display
            uri = doc.document.uri or "N/A"
            if len(uri) > 32:
                uri = "..." + uri[-29:]

            # Format debug values with proper precision
            c_rank = (
                f"{doc.combined_rank:.17f}" if doc.combined_rank is not None else "N/A"
            )
            v_rank = str(doc.vec_rank) if doc.vec_rank is not None else "N/A"
            fts_rank = str(doc.fts_rank) if doc.fts_rank is not None else "N/A"
            v_dist = (
                f"{doc.vec_distance:.6f}" if doc.vec_distance is not None else "N/A"
            )
            fts_score = f"{doc.fts_score:.6f}" if doc.fts_score is not None else "N/A"

            typer.echo(
                f"{idx:<3} {snippet:<55} {uri:<35} {c_rank:<33} {v_rank:<8} {fts_rank:<9} {v_dist:<18} {fts_score:<18}"
            )


class LegacyCompactFormatter(SearchResultFormatter):
    """Legacy compact formatter for backwards compatibility."""

    def format_results(self, results: List[DocumentResult], query: str) -> None:
        if not results:
            typer.echo("No documents found matching the query.")
            return

        typer.echo(f"Found {len(results)} documents:")

        # Clean simple table for normal view
        typer.echo(f"{'#':<3} {'Preview':<60} {'URI':<40}")
        typer.echo("─" * 105)

        for idx, doc in enumerate(results, 1):
            # Clean snippet display
            snippet = doc.snippet.replace("\n", " ").replace("\r", "")
            if len(snippet) > 57:
                snippet = snippet[:54] + "..."

            # Clean URI display
            uri = doc.document.uri or "N/A"
            if len(uri) > 37:
                uri = "..." + uri[-34:]

            typer.echo(f"{idx:<3} {snippet:<60} {uri:<40}")


def get_formatter(
    debug: bool = False, table_view: bool = False
) -> SearchResultFormatter:
    """Factory function to get the appropriate formatter."""
    if table_view:
        return TableDebugFormatter()
    elif debug:
        return ModernDetailedFormatter()
    else:
        return ModernCompactFormatter()
