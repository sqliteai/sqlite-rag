import sqlite3
import tempfile

import pytest

from database import Database
from settings import Settings


@pytest.fixture
def db_conn():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        settings = Settings(
            model_path_or_name="./capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
            db_path=tmp_db.name,
        )

    conn = sqlite3.connect(settings.db_path)
    Database.initialize(conn, settings)

    yield conn, settings

    conn.close()


@pytest.fixture
def db_settings() -> Settings:
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        settings = Settings(model_path_or_name="./capybarahermes-2.5-mistral-7b.Q4_K_M.gguf", db_path=tmp_db.name)
    return settings
