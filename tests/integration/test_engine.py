import random
import string

import pytest

from sqlite_rag.models.chunk import Chunk


class TestEngine:
    @pytest.mark.slow
    def test_stress_embedding_generation(self, engine):
        """Test embedding generation with a large number of chunks
        to not fail and to never generate duplicated embeddings."""

        def random_string(length=30):
            return "".join(
                random.choices(string.ascii_letters + string.digits + " ", k=length)
            )

        result_chunks = {}
        for i in range(1000):
            try:
                chunk = engine.generate_embedding([Chunk(content=random_string())])
                result_chunks[chunk[0].embedding.hex()] = chunk[0]
                assert len(result_chunks) == i + 1
            except Exception as e:
                pytest.fail(f"Embedding generation failed on chunk {i}: {e}")

        # Assert
        assert len(result_chunks) == 1000
