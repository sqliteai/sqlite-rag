from datetime import datetime

from attr import dataclass

from .chunk import Chunk


@dataclass
class Document:
    id: str | None = None
    content: str = ""
    uri: str | None = None
    metadata: dict = {}
    created_at: datetime | None = None
    updated_at: datetime | None = None

    chunks: list["Chunk"] = []

    vec_rank: float | None = None
    fts_rank: float | None = None
    combined_rank: float | None = None
    vec_distance: float | None = None
    fts_score: float | None = None

    def hash(self) -> str:
        """Generate a hash for the document content"""
        return str(hash(self.content))
