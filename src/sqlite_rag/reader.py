from pathlib import Path

from markitdown import MarkItDown, StreamInfo


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
            converter = MarkItDown()
            return converter.convert(path, stream_info=StreamInfo(charset='utf8')).text_content
        except Exception as exc:
            raise ValueError(f"Failed to parse file {path}") from exc

    @staticmethod
    def collect_files(path: Path, recursive: bool = False) -> list[Path]:
        """Collect files from the path, optionally recursively"""
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

        if path.is_file() and FileReader.is_supported(path):
            return [path]

        files_to_process = []
        if path.is_dir():
            if recursive:
                files_to_process = list(path.rglob("*"))
            else:
                files_to_process = list(path.glob("*"))

            files_to_process = [
                f
                for f in files_to_process
                if f.is_file() and FileReader.is_supported(f)
            ]

        return files_to_process
