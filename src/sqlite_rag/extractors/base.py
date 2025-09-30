from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple


class MetadataExtractor(ABC):
    """Base interface for metadata extractors."""

    @abstractmethod
    def extract(
        self, content: str, file_path: Optional[Path] = None
    ) -> Tuple[str, Dict]:
        """Extract metadata from content.

        Args:
            content: The raw content to extract metadata from
            file_path: Optional file path for context

        Returns:
            Tuple of (clean_content, metadata_dict)
        """

    @abstractmethod
    def supports_file_type(self, file_extension: str) -> bool:
        """Check if this extractor supports the given file type.

        Args:
            file_extension: File extension (e.g., '.md', '.pdf')

        Returns:
            True if this extractor can handle the file type
        """
