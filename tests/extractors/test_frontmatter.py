from sqlite_rag.extractors.frontmatter import FrontmatterExtractor


class TestFrontmatterExtractor:
    def test_extract_with_frontmatter(self):
        content = """---
title: Test Document
author: John Doe
---
# Heading 1
This is a test document.
"""
        extractor = FrontmatterExtractor()
        clean_content, metadata = extractor.extract(content)
        assert "title" in metadata
        assert metadata["title"] == "Test Document"
        assert "author" in metadata
        assert metadata["author"] == "John Doe"
        assert "# Heading 1" in clean_content
        assert "This is a test document." in clean_content

    def test_extract_without_frontmatter(self):
        content = """# Heading 1
This is a test document without frontmatter.
"""
        extractor = FrontmatterExtractor()
        clean_content, metadata = extractor.extract(content)
        assert metadata == {}
        assert "# Heading 1" in clean_content
        assert "This is a test document without frontmatter." in clean_content

    def test_supports_file_type(self):
        extractor = FrontmatterExtractor()
        assert extractor.supports_file_type(".md")
        assert extractor.supports_file_type(".MDX")
        assert extractor.supports_file_type(".txt")
        assert not extractor.supports_file_type(".pdf")
        assert not extractor.supports_file_type(".html")

    def test_extract_malformed_frontmatter(self):
        content = """---
title: Test Document
author John Doe
---
# Heading 1
"""
        extractor = FrontmatterExtractor()
        clean_content, metadata = extractor.extract(content)
        # Should return original content and empty metadata on failure
        assert metadata == {}
        assert content == clean_content
        assert "# Heading 1" in clean_content
