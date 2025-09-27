import pytest

from sqlite_rag.models.document import Document


class TestDocument:
    def test_get_title(self):
        doc = Document(content="Sample content", metadata={"title": "My Title"})
        assert doc.get_title() == "My Title"

    def test_get_title_no_title(self):
        doc = Document(content="Sample content", metadata={})
        assert doc.get_title() is None

    def test_set_generated_title(self):
        doc = Document(content="This is a test document.", metadata={})
        doc.set_generated_title()
        assert doc.get_title() == "This is a test document."

    def test_extract_document_title_with_heading(self):
        content = "# Document Title\nThis is the content."
        doc = Document(content=content, metadata={})
        assert doc.extract_document_title() == "Document Title"

    @pytest.mark.parametrize(
        "content,fallback,expected_title",
        [
            (
                "This is the first line.\nThis is the second line.",
                True,
                "This is the first line.",
            ),
            ("\n\nFirst non-empty line.\nAnother line.", True, "First non-empty line."),
            ("   \n   \n  Leading spaces line.", True, "Leading spaces line."),
            ("", True, None),
            ("\n\n", True, None),
            ("This is the only line.", False, None),
            ("\n\n", False, None),
        ],
    )
    def test_extract_document_title_without_heading(
        self, content, fallback, expected_title
    ):
        doc = Document(content=content, metadata={})
        assert (
            doc.extract_document_title(fallback_first_line=fallback) == expected_title
        )

    def test_extract_document_title_with_a_word(self):
        content = "---\n    \n  Leading spaces line with a word."
        doc = Document(content=content, metadata={})
        assert (
            doc.extract_document_title(fallback_first_line=True)
            == "Leading spaces line with a word."
        )
