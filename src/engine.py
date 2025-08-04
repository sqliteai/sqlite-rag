import sqlite3
from pathlib import Path

from models.chunk import Chunk
from settings import Settings


class Engine:
    def __init__(self, conn: sqlite3.Connection, settings: Settings):
        self._conn = conn
        self.settings = settings

    def load_model(self):
        """Load the model model from the specified path
        or download it from Hugging Face if not found."""

        model_path = Path(self.settings.model_path_or_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # model_path = self.settings.model_path_or_name
        # if not Path(self.settings.model_path_or_name).exists():
        #     # check if exists locally or try to download it from Hugging Face
        #     model_path = hf_hub_download(
        #         repo_id=self.settings.model_path_or_name,
        #         filename="model-q4_0.gguf",  # GGUF format
        #         cache_dir="./models"
        #     )

        self._conn.execute(
            f"SELECT llm_model_load('{self.settings.model_path_or_name}', '{self.settings.model_config}');"
        )

    def generate_embedding(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embedding for the given text."""
        cursor = self._conn.cursor()

        for chunk in chunks:
            cursor.execute("SELECT llm_embed_generate(?)", (chunk.content,))
            result = cursor.fetchone()

            if result is None:
                raise RuntimeError("Failed to generate embedding.")

            chunk.embedding = result[0]

        return chunks

    def quantize(self) -> None:
        """Quantitize stored vector for faster search via quantitize scan."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT vector_quantize('chunks', 'embedding');")

        self._conn.commit()
