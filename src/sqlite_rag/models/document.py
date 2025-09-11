import hashlib
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

    def hash(self) -> str:
        """Generate a hash for the document content using SHA-3 for maximum collision resistance"""
        return hashlib.sha3_256(self.content.encode()).hexdigest()
