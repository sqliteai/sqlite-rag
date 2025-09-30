from pathlib import Path
from typing import Dict, Optional, Tuple

from sqlite_rag.extractors.base import MetadataExtractor
from sqlite_rag.extractors.frontmatter import FrontmatterExtractor


class Extractor:
    extractors = [
        FrontmatterExtractor(),
    ]

    def get_extractor(self, file_extension: str) -> Optional[MetadataExtractor]:
        """Get the appropriate extractor based on file type."""
        for extractor in self.extractors:
            if extractor.supports_file_type(file_extension):
                return extractor

        return None

    def extract_metadata(self, content: str, file_path: Path) -> Tuple[str, Dict]:
        """Extract metadata and clean content based on file type.

        Args:
            content: Raw content to extract metadata from
            file_path: Path to the file for context

        Returns:
            Tuple of (clean_content, metadata_dict)
        """
        file_extension = file_path.suffix

        extractor = self.get_extractor(file_extension)
        if extractor:
            return extractor.extract(content, file_path)

        return content, {}
