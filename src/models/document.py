from datetime import datetime

from attr import dataclass

from models.chunk import Chunk


@dataclass
class Document:
    id: str | None = None
    content: str = ""
    uri: str | None = None
    metadata: dict = {}
    created_at: datetime | None = None
    updated_at: datetime | None = None

    chunks: list["Chunk"] = []

    def hash(self) -> str:
        """Generate a hash for the document content"""
        return str(hash(self.content))
