from pathlib import Path

from sqlite_rag.extractor import Extractor


class TestExtractor:
    def test_extract_metadata_from_md(self):
        extractor = Extractor()
        content = """---
title: Sample Document
author: Test Author
---
# Heading 1
This is a sample markdown document.
"""
        file_path = Path("sample.md")
        clean_content, metadata = extractor.extract_metadata(content, file_path)
        assert "title" in metadata
        assert metadata["title"] == "Sample Document"
        assert "author" in metadata
        assert metadata["author"] == "Test Author"
        assert "# Heading 1" in clean_content
        assert "This is a sample markdown document." in clean_content

    def test_no_extractor_for_unsupported_file(self):
        extractor = Extractor()
        content = "<html><body>This is HTML content.</body></html>"
        file_path = Path("sample.html")
        clean_content, metadata = extractor.extract_metadata(content, file_path)
        assert clean_content == content
        assert metadata == {}

    def test_get_extractor(self):
        extractor = Extractor()
        md_extractor = extractor.get_extractor(".md")
        assert md_extractor is not None
        assert md_extractor.supports_file_type(".md")

        html_extractor = extractor.get_extractor(".html")
        assert html_extractor is None
