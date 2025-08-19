from attr import dataclass


@dataclass
class Chunk:
    id: int | None = None
    document_id: int | None = None
    content: str = ""
    embedding: str | bytes = b""
