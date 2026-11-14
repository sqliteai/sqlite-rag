from pathlib import Path
from typing import Optional

from markitdown import MarkItDown, StreamInfo


class FileReader:
    extensions = [
        "c",
        "cpp",
        "css",
        "csv",
        "docx",
        "go",
        "h",
        "hpp",
        "html",
        "java",
        "js",
        "json",
        "kt",
        "md",
        "mdx",
        "mjs",
        "pdf",
        "php",
        "pptx",
        "py",
        "rb",
        "rs",
        "svelte",
        "swift",
        "ts",
        "tsx",
        "txt",
        "vue",
        "xml",
        "xlsx",
        "yaml",
        "yml",
    ]

    @staticmethod
    def _get_extensions_from_settings(settings):
        """Get supported extensions from settings object."""
        if settings and hasattr(settings, 'supported_extensions') and settings.supported_extensions:
            return {
                ext.strip().lstrip(".").lower()
                for ext in settings.supported_extensions.split(",")
                if ext.strip()
            }
        return set(FileReader.extensions)

    @staticmethod
    def is_supported(
        path: Path,
        settings=None,
        only_extensions: Optional[list[str]] = None,
        exclude_extensions: Optional[list[str]] = None,
    ) -> bool:
        """Check if the file extension is supported.

        Parameters:
            path (Path): The file path to check.
            only_extensions (Optional[list[str]]): If provided, only files with these extensions are considered.
            exclude_extensions (Optional[list[str]]): If provided, files with these extensions are excluded.
        """
        extension = path.suffix.lower().lstrip(".")

        # Get base supported extensions from settings or defaults
        supported_extensions = FileReader._get_extensions_from_settings(settings)
        exclude_set = set()

        # Only keep those that are in both lists
        if only_extensions:
            only_set = {ext.lower().lstrip(".") for ext in only_extensions}
            supported_extensions &= only_set

        if exclude_extensions:
            exclude_set = {ext.lower().lstrip(".") for ext in exclude_extensions}

        return extension in supported_extensions and extension not in exclude_set

    @staticmethod
    def parse_file(path: Path, max_document_size_bytes: Optional[int] = None) -> str:
        """Read the file and convert into Markdown text."""
        try:
            converter = MarkItDown()
            text = converter.convert(
                path, stream_info=StreamInfo(charset="utf8")
            ).text_content

            # Truncate text characters to max size if needed
            text = text.encode("utf-8", errors="ignore")
            if max_document_size_bytes:
                text = text[:max_document_size_bytes]

            return text.decode("utf-8", errors="ignore")

        except Exception as exc:
            raise ValueError(f"Failed to parse file {path}") from exc

    @staticmethod
    def collect_files(
        path: Path,
        recursive: bool = False,
        settings=None,
        only_extensions: Optional[list[str]] = None,
        exclude_extensions: Optional[list[str]] = None,
    ) -> list[Path]:
        """Collect files from the path, optionally recursively"""
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

        if path.is_file() and FileReader.is_supported(
            path, settings=settings, only_extensions=only_extensions, exclude_extensions=exclude_extensions
        ):
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
                if f.is_file()
                and FileReader.is_supported(
                    f, settings=settings, only_extensions=only_extensions, exclude_extensions=exclude_extensions
                )
            ]

        return files_to_process
