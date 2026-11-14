import tempfile
from pathlib import Path

import pytest

from sqlite_rag.reader import FileReader
from sqlite_rag.settings import Settings


class TestCustomFileExtensions:
    """Tests for custom file extensions support"""

    def test_default_extensions_fallback(self):
        """Test that FileReader uses its own default list when settings is empty"""
        # Empty settings should fallback to FileReader.extensions
        empty_settings = Settings(supported_extensions="")
        
        # .py is in default FileReader.extensions
        assert FileReader.is_supported(Path("test.py"), settings=empty_settings)
        
        # .c4 is NOT in default FileReader.extensions
        assert not FileReader.is_supported(Path("test.c4"), settings=empty_settings)

    def test_custom_extensions_override(self):
        """Test that custom extensions override defaults"""
        # Settings with custom extensions
        custom_settings = Settings(supported_extensions="c4,txt")
        
        # .c4 should be supported now
        assert FileReader.is_supported(Path("test.c4"), settings=custom_settings)
        
        # .txt should be supported (in custom list)
        assert FileReader.is_supported(Path("test.txt"), settings=custom_settings)
        
        # .py should NOT be supported (not in custom list, even though it's in defaults)
        assert not FileReader.is_supported(Path("test.py"), settings=custom_settings)

    def test_custom_extensions_with_only_default(self):
        """Test that custom list can contain just one extension"""
        single_ext_settings = Settings(supported_extensions="md")
        
        assert FileReader.is_supported(Path("test.md"), settings=single_ext_settings)
        assert not FileReader.is_supported(Path("test.py"), settings=single_ext_settings)
        assert not FileReader.is_supported(Path("test.txt"), settings=single_ext_settings)

    def test_extensions_with_leading_dots(self):
        """Test that extensions with leading dots are handled correctly"""
        settings_with_dots = Settings(supported_extensions=".py,.txt,.c4")
        
        assert FileReader.is_supported(Path("test.py"), settings=settings_with_dots)
        assert FileReader.is_supported(Path("test.txt"), settings=settings_with_dots)
        assert FileReader.is_supported(Path("test.c4"), settings=settings_with_dots)
        assert not FileReader.is_supported(Path("test.js"), settings=settings_with_dots)

    def test_extensions_case_insensitive(self):
        """Test that extension matching is case insensitive"""
        settings = Settings(supported_extensions="PY,TXT,c4")
        
        # All cases should work
        assert FileReader.is_supported(Path("test.PY"), settings=settings)
        assert FileReader.is_supported(Path("test.py"), settings=settings)
        assert FileReader.is_supported(Path("test.Py"), settings=settings)
        
        assert FileReader.is_supported(Path("test.TXT"), settings=settings)
        assert FileReader.is_supported(Path("test.txt"), settings=settings)
        
        assert FileReader.is_supported(Path("test.C4"), settings=settings)
        assert FileReader.is_supported(Path("test.c4"), settings=settings)

    def test_collect_files_with_custom_extensions(self):
        """Test that collect_files respects custom extensions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test.py").write_text("# Python file")
            (temp_path / "test.txt").write_text("# Text file")
            (temp_path / "test.c4").write_text("# LikeC4 file")
            (temp_path / "test.md").write_text("# Markdown file")
            
            # Custom settings only allow .c4
            custom_settings = Settings(supported_extensions="c4")
            
            files = FileReader.collect_files(temp_path, settings=custom_settings)
            
            assert len(files) == 1
            assert (temp_path / "test.c4") in files

    def test_settings_none_fallback_to_file_reader_defaults(self):
        """Test that None settings falls back to FileReader.extensions"""
        # When settings is None, should use FileReader.extensions
        assert FileReader.is_supported(Path("test.py"), settings=None)
        assert FileReader.is_supported(Path("test.txt"), settings=None)
        assert FileReader.is_supported(Path("test.md"), settings=None)
        
        # .c4 should not be supported when using defaults
        assert not FileReader.is_supported(Path("test.c4"), settings=None)