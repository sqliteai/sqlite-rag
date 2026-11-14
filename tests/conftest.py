import sqlite3
import tempfile
from collections.abc import Generator

import pytest

from sqlite_rag.chunker import Chunker
from sqlite_rag.database import Database
from sqlite_rag.engine import Engine
from sqlite_rag.sentence_splitter import SentenceSplitter
from sqlite_rag.settings import Settings


@pytest.fixture
def db_conn():
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db_path = tmp_db.name
    tmp_db.close()

    try:
        settings = Settings()

        conn = sqlite3.connect(tmp_db_path)
        conn.row_factory = sqlite3.Row

        Database.initialize(conn, settings)

        yield conn, settings

        conn.close()
    finally:
        import os
        try:
            os.unlink(tmp_db_path)
        except OSError:
            pass


@pytest.fixture
def engine(db_conn) -> Generator[Engine, None, None]:
    conn, settings = db_conn

    engine = Engine(
        conn,
        settings,
        chunker=Chunker(conn, settings),
        sentence_splitter=SentenceSplitter(),
    )
    engine.load_model()
    engine.quantize()
    engine.create_new_context()

    yield engine

    # Cleanup resources to prevent segfaults in Python 3.11
    # Must explicitly free resources before garbage collection
    try:
        engine.close()
    except Exception:
        pass
