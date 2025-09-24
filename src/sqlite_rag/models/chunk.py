from attr import dataclass


@dataclass
class Chunk:
    id: int | None = None
    document_id: int | None = None
    # The human readable content of the chunk
    # (not the representation of the embedding vector)
    content: str = ""
    embedding: str | bytes = b""

    prompt: str | None = None
    head_overlap_text: str = ""
    title: str | None = None

    def get_embedding_text(self) -> str:
        """Get the content used to generate the embedding from.
        It can be enriched with overlap text and prompt instructions,
        depending on the model preferences.
        """
        content = self.head_overlap_text + self.content

        if self.prompt:
            return self.prompt.format(title=self.title or "none", content=content)

        return self.content
