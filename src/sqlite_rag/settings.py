import json
import sqlite3
from dataclasses import asdict, dataclass, fields


@dataclass
class Settings:

    #
    # Model and embedding settings
    #

    model_path_or_name: str = (
        "./models/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"
    )
    model_config: str = "n_ctx=12000,pooling_type=last,normalize_embedding=1"

    vector_type: str = "FLOAT16"
    embedding_dim: int = 1024
    other_vector_config: str = "distance=cosine"  # e.g. distance=metric,other=value,...

    chunk_size: int = 12000
    # Token overlap between chunks
    chunk_overlap: int = 1200

    #
    # Search settings
    #

    # Whether to quantize the vector for faster search the full scan
    quantize_scan: bool = True
    # Load quantized vectors in memory for faster search
    quantize_preload: bool = False

    # Weights for combining FTS and vector search results
    weight_fts: float = 1.0
    weight_vec: float = 1.0


class SettingsManager:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id TEXT PRIMARY KEY,
                settings JSON NOT NULL
            );
        """
        )
        self.connection.commit()

    def load_settings(self) -> Settings | None:
        cursor = self.connection.cursor()

        cursor.execute("SELECT settings FROM settings LIMIT 1")
        row = cursor.fetchone()

        if not row:
            return None

        current_settings = json.loads(row[0])

        # Start from defaults, update with values from db (ignore extra keys)
        defaults = Settings()
        valid_keys = {f.name for f in fields(Settings)}
        filtered = {k: v for k, v in current_settings.items() if k in valid_keys}

        # Use defaults as base, update with valid properties
        settings_dict = {**asdict(defaults), **filtered}
        return Settings(**settings_dict)

    def store(self, settings: Settings):
        cursor = self.connection.cursor()

        settings_json = json.dumps(asdict(settings))

        # Upsert the settings
        cursor.execute(
            """
            INSERT INTO settings (id, settings)
            VALUES ('1', ?)
            ON CONFLICT(id) DO UPDATE SET settings = excluded.settings;
        """,
            (settings_json,),
        )

        self.connection.commit()
        return settings

    def has_critical_changes(
        self, new_settings: Settings, current_settings: Settings
    ) -> bool:
        """Check if the new settings have critical changes compared to the current settings."""
        return (
            new_settings.model_path_or_name != current_settings.model_path_or_name
            or new_settings.embedding_dim != current_settings.embedding_dim
            or new_settings.vector_type != current_settings.vector_type
        )
