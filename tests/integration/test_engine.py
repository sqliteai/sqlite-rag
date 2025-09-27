import random
import string

import pytest


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
                embedding = engine.generate_embedding(random_string())
                result_chunks[embedding.hex()] = embedding
                assert len(result_chunks) == i + 1
            except Exception as e:
                pytest.fail(f"Embedding generation failed on chunk {i}: {e}")

        # Assert
        assert len(result_chunks) == 1000
