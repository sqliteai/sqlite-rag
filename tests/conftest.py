import sqlite3
import tempfile

import pytest

from sqlite_rag.chunker import Chunker
from sqlite_rag.database import Database
from sqlite_rag.engine import Engine
from sqlite_rag.settings import Settings


@pytest.fixture
def db_conn():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        settings = Settings()

        conn = sqlite3.connect(tmp_db.name)
        conn.row_factory = sqlite3.Row

        Database.initialize(conn, settings)

        yield conn, settings

        conn.close()


@pytest.fixture
def engine(db_conn):
    conn, settings = db_conn

    engine = Engine(conn, settings, chunker=Chunker(conn, settings))
    engine.load_model()
    engine.quantize()

    return engine
