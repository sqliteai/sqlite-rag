import os
import tempfile
from pathlib import Path

from pytest import fixture
from typer.testing import CliRunner

from sqlite_rag.cli import app
from sqlite_rag.settings import Settings


@fixture
def temp_dir():
    """Change the current working directory in order to create
    the default database in a temporary location."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            os.chdir(original_cwd)


class TestCLI:
    def test_search_exact_match(self):
        # Use SQLiteRag to set up the test data directly
        with tempfile.TemporaryDirectory() as tmpdir:
            doc1_content = "The quick brown fox jumps over the lazy dog"
            doc2_content = (
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
            )

            runner = CliRunner()

            model_path = Path(Settings().model_path).absolute()

            # Change to the temporary directory so CLI finds the database
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # CWD has changed so the model must be referenced by absolute path
                result = runner.invoke(
                    app,
                    [
                        "set",
                        "--model-path-or-name",
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
                        "add-text",
                        doc1_content,
                    ],
                )
                assert result.exit_code == 0

                result = runner.invoke(
                    app,
                    [
                        "add-text",
                        doc2_content,
                    ],
                )
                assert result.exit_code == 0

                # Search
                result = runner.invoke(
                    app, ["search", doc1_content, "--debug", "--limit", "1"]
                )
            finally:
                os.chdir(original_cwd)

            # Assert CLI command executed successfully
            assert result.exit_code == 0
            assert "Found 1 documents" in result.stdout
            # For exact match with cosine distance, we expect distance close to 0.0
            assert "0.000000" in result.stdout or "0.00000" in result.stdout

    def test_set_settings(self, temp_dir):
        runner = CliRunner()

        model_path = "mypath/mymodel.gguf"

        result = runner.invoke(
            app,
            [
                "set",
                "--model-path-or-name",
                model_path,
                "--other-vector-options",
                "distance=L2",
            ],
        )
        assert result.exit_code == 0

        assert f"model_path: {model_path}" in result.stdout
        assert "other_vector_options: distance=L2" in result.stdout
