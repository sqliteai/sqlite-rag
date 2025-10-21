from dataclasses import dataclass, field

from sqlite_rag.models.sentence import Sentence


@dataclass
class Chunk:
    id: int | None = None
    document_id: int | None = None
    # The human readable content of the chunk
    # (it does not represent the embedding vector which
    # may be altered with prompt or overlap text)
    content: str = ""
    embedding: str | bytes = b""

    prompt: str | None = None
    head_overlap_text: str = ""
    title: str | None = None

    sentences: list[Sentence] = field(default_factory=list)

    def get_embedding_text(self) -> str:
        """Get the content used to generate the embedding from.
        It can be enriched with overlap text and prompt instructions,
        depending on the model preferences.
        """
        content = self.head_overlap_text + self.content

        if self.prompt:
            return self.prompt.format(title=self.title or "none", content=content)

        return self.content
