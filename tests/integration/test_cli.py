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
