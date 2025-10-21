from dataclasses import dataclass


@dataclass
class SentenceResult:
    id: int | None = None
    chunk_id: int | None = None

    content: str = ""

    rank: float | None = None
    distance: float | None = None

    start_offset: int | None = None
    end_offset: int | None = None
