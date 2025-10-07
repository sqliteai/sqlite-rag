import tempfile
from pathlib import Path

from typer.testing import CliRunner

from sqlite_rag.cli import app
from sqlite_rag.settings import Settings


class TestCLI:
    def test_search_exact_match(self):
        """Test adding documents and searching for an exact match."""
        doc1_content = "The quick brown fox jumps over the lazy dog"
        doc2_content = (
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
        )

        with tempfile.NamedTemporaryFile(suffix=".tempdb") as tmp_db:

            runner = CliRunner()

            model_path = Path(Settings().model_path).absolute()

            result = runner.invoke(
                app,
                [
                    "--database",
                    tmp_db.name,
                    "configure",
                    "--model-path",
                    str(model_path),
                    "--no-prompt-templates",
                    "--other-vector-options",
                    "distance=cosine",
                ],
            )
            assert result.exit_code == 0

            # Add
            result = runner.invoke(
                app,
                [
                    "--database",
                    tmp_db.name,
                    "add-text",
                    doc1_content,
                ],
            )
            assert result.exit_code == 0

            result = runner.invoke(
                app,
                [
                    "--database",
                    tmp_db.name,
                    "add-text",
                    doc2_content,
                ],
            )
            assert result.exit_code == 0

            # Search
            result = runner.invoke(
                app,
                [
                    "--database",
                    tmp_db.name,
                    "search",
                    doc1_content,
                    "--debug",
                    "--limit",
                    "1",
                ],
            )

            # Assert CLI command executed successfully
            assert result.exit_code == 0
            assert "Search Results (1 matches)" in result.stdout
            # For exact match with cosine distance, we expect distance close to 0.0
            assert "Vector: 0.000000" in result.stdout or "0.00000" in result.stdout

    def test_set_settings(self):
        with tempfile.NamedTemporaryFile(suffix=".tempdb") as tmp_db:
            runner = CliRunner()

            model_path = "mypath/mymodel.gguf"

            result = runner.invoke(
                app,
                [
                    "--database",
                    tmp_db.name,
                    "configure",
                    "--model-path",
                    model_path,
                    "--other-vector-options",
                    "distance=L2",
                ],
            )
            assert result.exit_code == 0

            assert f"model_path: {model_path}" in result.stdout
            assert "other_vector_options: distance=L2" in result.stdout

    def test_change_database_path(self):
        with tempfile.NamedTemporaryFile(suffix=".tempdb") as tmp_db:
            runner = CliRunner()

            result = runner.invoke(
                app,
                ["--database", tmp_db.name, "settings"],
            )
            assert result.exit_code == 0

            assert f"Database: {tmp_db.name}" in result.stdout

    def test_add_with_exclude_extensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "file1.txt").write_text("This is a text file.")
            (Path(tmp_dir) / "file2.md").write_text("# This is a markdown file.")
            (Path(tmp_dir) / "file3.py").write_text("print('Hello, world!')")
            (Path(tmp_dir) / "file4.js").write_text("console.log('Hello, world!');")

            with tempfile.NamedTemporaryFile(suffix=".tempdb") as tmp_db:
                runner = CliRunner()

                result = runner.invoke(
                    app,
                    ["--database", tmp_db.name, "add", tmp_dir, "--exclude", "py,js"],
                )
                assert result.exit_code == 0

                # Check that only .txt and .md files were added
                assert "Processing 2 files" in result.stdout
                assert "file1.txt" in result.stdout
                assert "file2.md" in result.stdout
                assert "file3.py" not in result.stdout
                assert "file4.js" not in result.stdout

    def test_add_with_only_extensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "file1.txt").write_text("This is a text file.")
            (Path(tmp_dir) / "file2.md").write_text("# This is a markdown file.")
            (Path(tmp_dir) / "file3.py").write_text("print('Hello, world!')")
            (Path(tmp_dir) / "file4.js").write_text("console.log('Hello, world!');")

            with tempfile.NamedTemporaryFile(suffix=".tempdb") as tmp_db:
                runner = CliRunner()

                result = runner.invoke(
                    app,
                    [
                        "--database",
                        tmp_db.name,
                        "add",
                        tmp_dir,
                        "--only",
                        "md,txt",
                    ],
                )
                assert result.exit_code == 0

                # Check that only .txt and .md files were added
                assert "Processing 2 files" in result.stdout
                assert "file1.txt" in result.stdout
                assert "file2.md" in result.stdout
                assert "file3.py" not in result.stdout
                assert "file4.js" not in result.stdout

    def test_add_with_only_and_exclude_extensions_are_normilized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "file1.txt").write_text("This is a text file.")
            (Path(tmp_dir) / "file2.md").write_text("# This is a markdown file.")
            (Path(tmp_dir) / "file3.py").write_text("print('Hello, world!')")

            with tempfile.NamedTemporaryFile(suffix=".tempdb") as tmp_db:
                runner = CliRunner()

                result = runner.invoke(
                    app,
                    [
                        "--database",
                        tmp_db.name,
                        "add",
                        tmp_dir,
                        "--only",
                        ".md, .txt,py",
                        "--exclude",
                        ".py ",  # wins over --only
                    ],
                )
                assert result.exit_code == 0

                # Check that only .txt and .md files were added
                assert "Processing 2 files" in result.stdout
                assert "file1.txt" in result.stdout
                assert "file2.md" in result.stdout
                assert "file3.py" not in result.stdout
