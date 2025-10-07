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
        unsupported_extensions = ["exe", "bin", "jpg", "png"]

        for ext in FileReader.extensions:
            assert FileReader.is_supported(Path(f"test.{ext}"))

        for ext in unsupported_extensions:
            assert not FileReader.is_supported(Path(f"test.{ext}"))

    def test_is_supported_with_only_extensions(self):
        """Test is_supported with only_extensions parameter"""
        # Test with only_extensions - should only allow specified extensions
        assert FileReader.is_supported(Path("test.py"), only_extensions=["py", "js"])
        assert FileReader.is_supported(Path("test.js"), only_extensions=["py", "js"])
        assert not FileReader.is_supported(
            Path("test.txt"), only_extensions=["py", "js"]
        )
        assert not FileReader.is_supported(
            Path("test.md"), only_extensions=["py", "js"]
        )

        # Test with dots in extensions (should be normalized)
        assert FileReader.is_supported(Path("test.py"), only_extensions=[".py", ".js"])
        assert FileReader.is_supported(Path("test.js"), only_extensions=[".py", ".js"])

        # Test case insensitive
        assert FileReader.is_supported(Path("test.py"), only_extensions=["PY", "JS"])
        assert FileReader.is_supported(Path("test.JS"), only_extensions=["py", "js"])

    def test_is_supported_with_exclude_extensions(self):
        """Test is_supported with exclude_extensions parameter"""
        # Test basic exclusion - py files should be excluded
        assert not FileReader.is_supported(Path("test.py"), exclude_extensions=["py"])
        assert FileReader.is_supported(Path("test.js"), exclude_extensions=["py"])
        assert FileReader.is_supported(Path("test.txt"), exclude_extensions=["py"])

        # Test with dots in extensions (should be normalized)
        assert not FileReader.is_supported(Path("test.py"), exclude_extensions=[".py"])
        assert FileReader.is_supported(Path("test.js"), exclude_extensions=[".py"])

        # Test case insensitive
        assert not FileReader.is_supported(Path("test.py"), exclude_extensions=["PY"])
        assert not FileReader.is_supported(Path("test.PY"), exclude_extensions=["py"])

        # Test multiple exclusions
        assert not FileReader.is_supported(
            Path("test.py"), exclude_extensions=["py", "js"]
        )
        assert not FileReader.is_supported(
            Path("test.js"), exclude_extensions=["py", "js"]
        )
        assert FileReader.is_supported(
            Path("test.txt"), exclude_extensions=["py", "js"]
        )

    def test_is_supported_with_only_and_exclude_extensions(self):
        """Test is_supported with both only_extensions and exclude_extensions"""
        # Include py and js, but exclude py - should only allow js
        assert not FileReader.is_supported(
            Path("test.py"), only_extensions=["py", "js"], exclude_extensions=["py"]
        )
        assert FileReader.is_supported(
            Path("test.js"), only_extensions=["py", "js"], exclude_extensions=["py"]
        )
        assert not FileReader.is_supported(
            Path("test.txt"), only_extensions=["py", "js"], exclude_extensions=["py"]
        )

        # Include py, txt, md, but exclude md - should only allow py and txt
        assert FileReader.is_supported(
            Path("test.py"),
            only_extensions=["py", "txt", "md"],
            exclude_extensions=["md"],
        )
        assert FileReader.is_supported(
            Path("test.txt"),
            only_extensions=["py", "txt", "md"],
            exclude_extensions=["md"],
        )
        assert not FileReader.is_supported(
            Path("test.md"),
            only_extensions=["py", "txt", "md"],
            exclude_extensions=["md"],
        )
        assert not FileReader.is_supported(
            Path("test.js"),
            only_extensions=["py", "txt", "md"],
            exclude_extensions=["md"],
        )

    def test_is_supported_with_unsupported_extensions_in_only(self):
        """Test that only_extensions can't add unsupported extensions"""
        # .exe is not in FileReader.extensions, so should not be supported even if in only_extensions
        assert not FileReader.is_supported(
            Path("test.exe"), only_extensions=["exe", "py"]
        )
        assert FileReader.is_supported(Path("test.py"), only_extensions=["exe", "py"])

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

    def test_parse_file_with_max_document_size_bytes(self):
        """Test that FileReader truncates content when max_document_size_bytes is specified"""
        long_content = "This is a very long document." * 100  # ~3000 chars
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(long_content.encode("utf-8"))
            temp_file_path = f.name

        max_size_bytes = 50
        content = FileReader.parse_file(
            Path(temp_file_path), max_document_size_bytes=max_size_bytes
        )

        # Content should be truncated to max_size bytes
        assert len(content.encode("utf-8")) <= max_size_bytes
        assert content.startswith("This is a very long document.")

        # Test without size limit
        full_content = FileReader.parse_file(Path(temp_file_path))
        assert len(full_content) == len(long_content)
        assert full_content == long_content
