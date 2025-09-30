from pathlib import Path
from typing import Dict, Optional, Tuple

import frontmatter

from sqlite_rag.extractors.base import MetadataExtractor


class FrontmatterExtractor(MetadataExtractor):
    """Extracts frontmatter from markdown files."""

    def extract(
        self, content: str, file_path: Optional[Path] = None
    ) -> Tuple[str, Dict]:
        """Extract frontmatter from markdown content."""
        try:
            post = frontmatter.loads(content)
            clean_content = post.content
            metadata = dict(post.metadata)
            return clean_content, metadata
        except Exception:
            # If frontmatter parsing fails, return original content
            return content, {}

    def supports_file_type(self, file_extension: str) -> bool:
        """Support markdown files."""
        return file_extension.lower() in [".md", ".mdx", ".txt"]
