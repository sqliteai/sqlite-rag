from pathlib import Path

from markitdown import MarkItDown


class FileReader:
    extensions = [
        ".c",
        ".cpp",
        ".css",
        ".csv",
        ".docx",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".kt",
        ".md",
        ".mdx",
        ".mjs",
        ".pdf",
        ".php",
        ".pptx",
        ".py",
        ".rb",
        ".rs",
        ".svelte",
        ".swift",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".xlsx",
        ".yaml",
        ".yml",
    ]

    @staticmethod
    def is_supported(path: Path) -> bool:
        """Check if the file extension is supported"""
        return path.suffix.lower() in FileReader.extensions

    @staticmethod
    def parse_file(path: Path) -> str:
        try:
            reader = MarkItDown()
            return reader.convert(path).text_content
        except Exception as exc:
            raise ValueError(f"Failed to parse file {path}") from exc
