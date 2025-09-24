from sqlite_rag.models.chunk import Chunk


class TestChunk:
    def test_get_embedding_text_without_prompt(self):
        expected_content = "This is the main content."
        chunk = Chunk(
            content=expected_content,
            head_overlap_text="This is the head overlap.",
            title="Sample Title",
        )

        assert chunk.get_embedding_text() == expected_content

    def test_get_embedding_text_with_prompt(self):
        prompt_template = "Title: {title}\nContent: {content}"
        main_content = "This is the main content."
        head_overlap = "This is the head overlap."
        title = "Sample Title"

        chunk = Chunk(
            content=main_content,
            head_overlap_text=head_overlap,
            prompt=prompt_template,
            title=title,
        )

        expected_content = prompt_template.format(
            title=title, content=head_overlap + main_content
        )
        assert chunk.get_embedding_text() == expected_content

    def test_get_embedding_text_with_prompt_no_title(self):
        prompt_template = "Title: {title}\nContent: {content}"
        main_content = "This is the main content."
        head_overlap = "This is the head overlap."

        chunk = Chunk(
            content=main_content,
            head_overlap_text=head_overlap,
            prompt=prompt_template,
            title=None,
        )

        expected_content = prompt_template.format(
            title="none", content=head_overlap + main_content
        )
        assert chunk.get_embedding_text() == expected_content

    def test_get_embedding_text_empty_head_overlap(self):
        prompt_template = "Title: {title}\nContent: {content}"
        main_content = "This is the main content."
        title = "Sample Title"

        chunk = Chunk(
            content=main_content,
            head_overlap_text="",
            prompt=prompt_template,
            title=title,
        )

        expected_content = prompt_template.format(title=title, content=main_content)
        assert chunk.get_embedding_text() == expected_content
