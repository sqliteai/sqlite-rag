from attr import dataclass


@dataclass
class Chunk:
    id: int | None = None
    document_id: int | None = None
    content: str = ""
    embedding: str | bytes = b""
    core_start_pos: int = 0

    title: str | None = None
