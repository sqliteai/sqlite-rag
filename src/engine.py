import sqlite3
from anyio import Path
from huggingface_hub import hf_hub_download
from repository import Repository
from settings import Settings


class Engine:
    def __init__(self, conn: sqlite3.Connection, settings: Settings):
        self._conn = conn
        self.settings = settings

    def load_model(self):
        """Load the model model from the specified path
        or download it from Hugging Face if not found."""
        model_path = self.settings.model_path_or_name
        # if not Path(self.settings.model_path_or_name).exists():
        #     # check if exists locally or try to download it from Hugging Face
        #     model_path = hf_hub_download(
        #         repo_id=self.settings.model_path_or_name,
        #         filename="model-q4_0.gguf",  # GGUF format
        #         cache_dir="./models"
        #     )

        self.repository.load_model(model_path)

    