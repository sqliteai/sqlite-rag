from engine import Engine
from models.chunk import Chunk


class TestEngine:
    def test_generate_embedding(self, db_conn):
        conn, settings = db_conn
        engine = Engine(conn, settings)
        engine.load_model()

        # Create a sample chunk
        chunk = Chunk(content="This is a test chunk for embedding generation.")

        # Generate embedding
        result_chunks = engine.generate_embedding([chunk])

        assert len(result_chunks) == 1
        assert result_chunks[0].embedding is not None
        assert isinstance(result_chunks[0].embedding, bytes)
