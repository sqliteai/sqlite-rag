from sqlite_rag.formatters import (
    BoxedFormatter,
    TableDebugFormatter,
    get_formatter,
)
from sqlite_rag.models.document import Document
from sqlite_rag.models.document_result import DocumentResult
from sqlite_rag.models.sentence_result import SentenceResult


class TestGetFormatter:
    """Test the get_formatter factory function."""

    def test_get_formatter_default(self):
        """Test getting formatter with default parameters."""
        formatter = get_formatter()
        assert isinstance(formatter, BoxedFormatter)
        assert formatter.show_debug is False

    def test_get_formatter_debug(self):
        """Test getting formatter with debug=True."""
        formatter = get_formatter(debug=True)
        assert isinstance(formatter, BoxedFormatter)
        assert formatter.show_debug is True

    def test_get_formatter_table_view(self):
        """Test getting table formatter."""
        formatter = get_formatter(table_view=True)
        assert isinstance(formatter, TableDebugFormatter)

    def test_get_formatter_table_view_takes_precedence(self):
        """Test that table_view takes precedence over debug."""
        formatter = get_formatter(debug=True, table_view=True)
        assert isinstance(formatter, TableDebugFormatter)
        # Table formatter doesn't have show_debug attribute


class TestSearchResultFormatter:
    """Test base SearchResultFormatter methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = BoxedFormatter()

    def test_get_file_icon_python(self):
        """Test getting icon for Python files."""
        assert self.formatter._get_file_icon("test.py") == "🐍"
        assert self.formatter._get_file_icon("test.pyx") == "🐍"

    def test_get_file_icon_javascript(self):
        """Test getting icon for JavaScript/TypeScript files."""
        assert self.formatter._get_file_icon("test.js") == "⚡"
        assert self.formatter._get_file_icon("test.ts") == "⚡"
        assert self.formatter._get_file_icon("test.jsx") == "⚡"
        assert self.formatter._get_file_icon("test.tsx") == "⚡"

    def test_get_file_icon_markdown(self):
        """Test getting icon for Markdown files."""
        assert self.formatter._get_file_icon("README.md") == "📄"
        assert self.formatter._get_file_icon("doc.markdown") == "📄"

    def test_get_file_icon_case_insensitive(self):
        """Test that file icon detection is case insensitive."""
        assert self.formatter._get_file_icon("TEST.PY") == "🐍"
        assert self.formatter._get_file_icon("Test.Js") == "⚡"

    def test_get_file_icon_empty_uri(self):
        """Test getting icon for empty URI."""
        assert self.formatter._get_file_icon("") == "📝"

    def test_get_file_icon_unknown_extension(self):
        """Test getting default icon for unknown extensions."""
        assert self.formatter._get_file_icon("test.xyz") == "📄"

    def test_clean_and_wrap_snippet_basic(self):
        """Test basic snippet cleaning and wrapping."""
        snippet = "This is a simple test snippet."
        result = self.formatter._clean_and_wrap_snippet(snippet, width=30)
        assert len(result) > 0
        assert all(len(line) <= 30 for line in result)

    def test_clean_and_wrap_snippet_removes_newlines(self):
        """Test that newlines and carriage returns are removed."""
        snippet = "Line 1\nLine 2\r\nLine 3"
        result = self.formatter._clean_and_wrap_snippet(snippet)
        combined = " ".join(result)
        assert "\n" not in combined
        assert "\r" not in combined
        assert "Line 1 Line 2 Line 3" == combined

    def test_clean_and_wrap_snippet_truncates_long_text(self):
        """Test that long snippets are truncated."""
        snippet = "A" * 500
        result = self.formatter._clean_and_wrap_snippet(snippet, max_length=100)
        combined = "".join(result)
        assert len(combined) <= 103  # 100 + "..."
        assert combined.endswith("...")

    def test_format_uri_display_basic(self):
        """Test basic URI formatting."""
        uri_display = self.formatter._format_uri_display(
            "path/to/file.py", "🐍", max_width=50
        )
        assert uri_display == "🐍 path/to/file.py"

    def test_format_uri_display_truncates_long_uri(self):
        """Test that long URIs are truncated."""
        long_uri = "very/long/path/" * 10 + "file.py"
        uri_display = self.formatter._format_uri_display(long_uri, "🐍", max_width=50)
        assert len(uri_display) <= 50
        assert uri_display.startswith("🐍 ...")

    def test_format_uri_display_empty_uri(self):
        """Test formatting empty URI."""
        assert self.formatter._format_uri_display("", "🐍") == ""


class TestBoxedFormatter:
    """Test BoxedFormatter functionality."""

    def test_init_default(self):
        """Test BoxedFormatter initialization with default parameters."""
        formatter = BoxedFormatter()
        assert formatter.show_debug is False

    def test_init_with_debug(self):
        """Test BoxedFormatter initialization with debug enabled."""
        formatter = BoxedFormatter(show_debug=True)
        assert formatter.show_debug is True

    def test_format_results_empty(self, mocker):
        """Test formatting with empty results."""
        formatter = BoxedFormatter()
        mock_echo = mocker.patch("typer.echo")
        formatter.format_results([], "test query")
        mock_echo.assert_called_once_with("No documents found matching the query.")

    def test_format_results_with_results(self, mocker):
        """Test formatting with actual results."""
        doc = Document(uri="test.py", content="test content")
        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content="This is test content.",
            combined_rank=0.95,
            vec_rank=1,
            fts_rank=2,
            vec_distance=0.1,
            fts_score=5.0,
        )

        formatter = BoxedFormatter()
        mock_echo = mocker.patch("typer.echo")
        formatter.format_results([result], "test query")
        # Should print header, result box, and empty line
        assert mock_echo.call_count > 3
        # Check that it prints the search results header
        first_call = mock_echo.call_args_list[0][0][0]
        assert "Search Results" in first_call
        assert "1 matches" in first_call

    def test_format_results_with_debug(self, mocker):
        """Test formatting with debug information."""
        doc = Document(uri="test.py", content="test content")
        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content="This is test content.",
            combined_rank=0.95,
            vec_rank=1,
            fts_rank=2,
            vec_distance=0.123456,
            fts_score=5.678901,
        )

        formatter = BoxedFormatter(show_debug=True)
        mock_echo = mocker.patch("typer.echo")
        formatter.format_results([result], "test query")
        # Check that debug info is printed
        output = "\n".join(
            [
                str(call.args[0]) if call.args else ""
                for call in mock_echo.call_args_list
            ]
        )
        assert "Combined:" in output
        assert "Vector:" in output
        assert "FTS:" in output

    def test_format_results_with_sentences_in_debug_mode(self, mocker):
        """Test formatting with sentence details in debug mode."""
        doc = Document(uri="test.py", content="test content")
        sentences = [
            SentenceResult(
                id=1,
                chunk_id=1,
                content="First sentence.",
                rank=1,
                distance=0.1,
                start_offset=0,
                end_offset=15,
            ),
            SentenceResult(
                id=2,
                chunk_id=1,
                content="Second sentence.",
                rank=2,
                distance=0.2,
                start_offset=16,
                end_offset=32,
            ),
        ]
        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content="First sentence. Second sentence.",
            combined_rank=0.95,
            sentences=sentences,
        )

        formatter = BoxedFormatter(show_debug=True)
        mock_echo = mocker.patch("typer.echo")
        formatter.format_results([result], "test query")
        output = "\n".join(
            [
                str(call.args[0]) if call.args else ""
                for call in mock_echo.call_args_list
            ]
        )
        assert "Sentences:" in output

    def test_format_results_without_sentences_in_non_debug_mode(self, mocker):
        """Test that sentences are not shown in non-debug mode."""
        doc = Document(uri="test.py", content="test content")
        sentences = [
            SentenceResult(
                id=1,
                chunk_id=1,
                content="First sentence.",
                rank=1,
                distance=0.1,
                start_offset=0,
                end_offset=15,
            ),
        ]
        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content="First sentence.",
            combined_rank=0.95,
            sentences=sentences,
        )

        formatter = BoxedFormatter(show_debug=False)
        mock_echo = mocker.patch("typer.echo")
        formatter.format_results([result], "test query")
        output = "\n".join(
            [
                str(call.args[0]) if call.args else ""
                for call in mock_echo.call_args_list
            ]
        )
        assert "Sentences:" not in output


class TestTableDebugFormatter:
    """Test TableDebugFormatter functionality."""

    def test_format_results_empty(self, mocker):
        """Test table formatting with empty results."""
        formatter = TableDebugFormatter()
        mock_echo = mocker.patch("typer.echo")
        formatter.format_results([], "test query")
        mock_echo.assert_called_once_with("No documents found matching the query.")

    def test_format_results_with_results(self, mocker):
        """Test table formatting with actual results."""
        doc = Document(uri="test.py", content="test content")
        result = DocumentResult(
            document=doc,
            chunk_id=1,
            chunk_content="This is test content.",
            combined_rank=0.95,
            vec_rank=1,
            fts_rank=2,
            vec_distance=0.1,
            fts_score=5.0,
        )

        formatter = TableDebugFormatter()
        mock_echo = mocker.patch("typer.echo")
        formatter.format_results([result], "test query")
        # Should print header, table header, separator, and row
        assert mock_echo.call_count >= 4
        # Check that headers are printed
        output = "\n".join([str(call[0][0]) for call in mock_echo.call_args_list])
        assert "Preview" in output
        assert "URI" in output
        assert "C.Rank" in output
