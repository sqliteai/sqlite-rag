import tempfile
from pathlib import Path

from sqlite_rag.reader import FileReader


class TestFileReader:
    def test_collect_files_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            files = FileReader.collect_files(Path(temp_dir), recursive=False)
            assert len(files) == 0

    def test_collect_files_single_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is a test document.")
            temp_file_path = f.name

        files = FileReader.collect_files(Path(temp_file_path), recursive=False)
        assert len(files) == 1
        assert Path(temp_file_path) in files

    def test_collect_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            file1 = temp_dir / "test1.txt"
            file2 = temp_dir / "test2.txt"
            file1.write_text("This is the first test document.")
            file2.write_text("This is the second test document.")

            files = FileReader.collect_files(temp_dir, recursive=False)
            assert len(files) == 2
            assert file1 in files
            assert file2 in files

    def test_collect_ignore_unsupported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sub_dir = Path(temp_dir) / "subdir"
            sub_dir.mkdir()
            file1 = Path(temp_dir) / "test1.txt"
            file2 = sub_dir / "test2.unsupported"
            file1.write_text("This is a test document.")
            file2.write_text("This is an unsupported file type.")

            files = FileReader.collect_files(Path(temp_dir), recursive=True)
            assert len(files) == 1
            assert file1 in files
            assert file2 not in files

    def test_collect_files_recursive_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sub_dir = Path(temp_dir) / "subdir"
            sub_dir.mkdir()
            file1 = Path(temp_dir) / "file1.txt"
            file2 = sub_dir / "file2.txt"

            file1.write_text("This is the first test document.")
            file2.write_text("This is the second test document.")

            files = FileReader.collect_files(Path(temp_dir), recursive=True)
            assert len(files) == 2
            assert file1 in files
            assert file2 in files

    def test_is_supported(self):
        unsupported_extensions = [".exe", ".bin", ".jpg", ".png"]

        for ext in FileReader.extensions:
            assert FileReader.is_supported(Path(f"test{ext}"))

        for ext in unsupported_extensions:
            assert not FileReader.is_supported(Path(f"test{ext}"))

    def test_parse_html_into_markdown(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(b"<html><body><h1>This is a test markdown file.</h1></body></html>")

        content = FileReader.parse_file(Path(f.name))
        assert "# This is a test markdown file." in content

    def test_markItDown_file_with_unicode_content(self):
        """Test that FileReader can handle UTF-8 files with Unicode characters like `±`"""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(
                "<html><body><h1>This is a document with a Unicode character: ±</h1></body></html>".encode(
                    "utf-8"
                )
            )

        # This should not raise a UnicodeDecodeError if MarkItDown's PlainTextConverter
        # is trying to decode as ASCII instead of UTF-8
        content = FileReader.parse_file(Path(f.name))
        assert "# This is a document with a Unicode character: ±" in content
