from dataclasses import dataclass


@dataclass
class Sentence:
    id: int | None = None
    content: str = ""
    embedding: str | bytes = b""
    sequence: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
