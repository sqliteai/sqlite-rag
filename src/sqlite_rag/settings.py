import json
import sqlite3
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Optional


@dataclass
class Settings:

    #
    # Model and embedding settings
    #

    model_path: str = (
        "./models/unsloth/embeddinggemma-300m-GGUF/embeddinggemma-300M-Q8_0.gguf"
    )
    # See: https://github.com/sqliteai/sqlite-ai/blob/main/API.md#llm_model_loadpath-text-options-text
    model_options: str = ""
    # See: https://github.com/sqliteai/sqlite-ai/blob/main/API.md#llm_context_createoptions-text
    model_context_options: str = (
        "generate_embedding=1,normalize_embedding=1,pooling_type=mean,embedding_type=INT8"
    )

    # Allow the sqlite-ai extension to use the GPU
    # See: https://github.com/sqliteai/sqlite-ai
    use_gpu = False

    vector_type: str = "INT8"
    embedding_dim: int = 768
    other_vector_options: str = (
        "distance=cosine"  # e.g. distance=metric,other=value,...
    )

    # It includes the overlap size and the prompt template length
    chunk_size: int = 512
    # Tokens overlap between chunks
    chunk_overlap: int = 61

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

    #
    # Prompt templates
    # Some models are trained to work better with specific prompts
    # depending on the task. For example, Gemma models work better
    # when the prompt includes a task description.
    # More: https://huggingface.co/unsloth/embeddinggemma-300m-GGUF#prompt-instructions
    #

    use_prompt_templates: bool = True

    # Template to index documents for retrieval, use `{title}` with the title or the string `"none"`
    prompt_template_retrieval_document: str = "title: {title} | text: {content}"
    prompt_template_retrieval_query: str = "task: search result | query: {content}"

    #
    # Index settings
    #

    # Maximum size of a document to process (in bytes)
    max_document_size_bytes: int = 5 * 1024 * 1024  # 5 MB
    # Zero means no limit
    max_chunks_per_document: int = 1000


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

    def configure(
        self, settings: Optional[dict[str, Any]], force: bool = False
    ) -> Settings:
        """Load, initialize or update settings.

        If no settings are provided, load the last used settings or use defaults.
        If settings are provided, check for critical changes and update them.

        Args:
            settings: A dictionary of settings to update.
            force: If True, skip critical changes check.
        """
        current_settings = self.load_settings()
        if current_settings:
            if settings:
                new_settings = replace(current_settings, **settings)

                has_critical = self.has_critical_changes(new_settings, current_settings)

                if has_critical:
                    if force:
                        print(
                            "Warning: Critical settings changes detected. Forcing update."
                        )
                    else:
                        raise ValueError(
                            "Critical settings changes detected. Please force the settings update or reset the database."
                        )
                # Update new settings
                current_settings = self.store(new_settings)
        elif settings:
            # Store initial settings with customs
            new_settings = replace(Settings(), **settings)
            current_settings = self.store(new_settings)
        else:
            # Store default settings
            new_settings = Settings()
            current_settings = self.store(new_settings)

        return current_settings

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
            new_settings.model_path != current_settings.model_path
            or new_settings.embedding_dim != current_settings.embedding_dim
            or new_settings.vector_type != current_settings.vector_type
        )
