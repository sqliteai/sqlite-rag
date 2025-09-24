import hashlib
import re
from datetime import datetime
from typing import Optional

from attr import dataclass

from .chunk import Chunk


@dataclass
class Document:
    GENERATED_TITLE_MAX_CHARS = 100

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

    def get_title(self) -> Optional[str]:
        """Extract title from metadata if available"""
        if self.metadata and "title" in self.metadata:
            return self.metadata["title"]
        if self.metadata and "generated" in self.metadata:
            if "title" in self.metadata["generated"]:
                return self.metadata["generated"]["title"]

        return None

    def set_generated_title(self):
        """Set a generated title in metadata"""
        if "generated" not in self.metadata:
            self.metadata["generated"] = {}
        self.metadata["generated"]["title"] = self.extract_document_title(
            fallback_first_line=True
        )

    def extract_document_title(self, fallback_first_line: bool = False) -> str | None:
        """Extract title from markdown content.

        Args:
            text: The markdown text to extract the title from.
            fallback_first_line: If True, use the first non-empty line as title if no heading is found.
        """
        # Look for first level-1 heading
        match = re.search(r"^# (.+)$", self.content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fallback: first non-empty line
        if fallback_first_line:
            for line in self.content.splitlines():
                line = line.strip()
                if line:
                    return line[: self.GENERATED_TITLE_MAX_CHARS]

        return None
