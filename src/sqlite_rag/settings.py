import json
import sqlite3
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Optional


@dataclass
class Settings:
    """Runtime configuration for the RAG pipeline."""

    #
    # Model and embedding settings
    #

    model_path: str = field(
        default=(
            "./models/unsloth/embeddinggemma-300m-GGUF/" "embeddinggemma-300M-Q8_0.gguf"
        ),
        metadata={"help": "Path to the embedding model file (.gguf)"},
    )
    # See: https://github.com/sqliteai/sqlite-ai/blob/main/API.md#llm_model_loadpath-text-options-text
    other_model_options: str = field(
        default="",
        metadata={"help": "Additional options for the embedding model loader"},
    )

    # See: https://github.com/sqliteai/sqlite-ai/blob/main/API.md#llm_context_createoptions-text
    other_model_context_options: str = field(
        default="",
        metadata={"help": "Extra parameters for embedding context creation"},
    )

    # How the model pools token embeddings into a single embedding
    # Options: "mean", "max", "min", "last", "first"
    pooling_type: str = field(
        default="mean",
        metadata={"help": "Pooling strategy for combining token embeddings"},
    )

    # Allow the sqlite-ai extension to use the GPU
    # See: https://github.com/sqliteai/sqlite-ai
    use_gpu: bool = field(
        default=False,
        metadata={
            "help": "Allow sqlite-ai extension to use the GPU",
            "cli_name": "use-gpu",
        },
    )

    # Best use the number of CPU available
    n_thread: int = field(
        default=4,
        metadata={"help": "Number of CPU threads used for embeddings"},
    )

    vector_type: str = field(
        default="INT8",
        metadata={"help": "Vector storage type (e.g. INT8, FLOAT16, FLOAT32)"},
    )
    embedding_dim: int = field(
        default=768,
        metadata={"help": "Dimension of each embedding vector"},
    )

    other_vector_options: str = field(
        default="distance=cosine",
        metadata={"help": "Extra vector options in key=value format"},
    )

    # It includes the overlap size and the prompt template length
    chunk_size: int = field(
        default=2048,
        metadata={"help": "Token budget per chunk (overlap + prompt template)"},
    )
    # Tokens overlap between chunks
    chunk_overlap: int = field(
        default=256,
        metadata={"help": "Number of tokens shared between consecutive chunks"},
    )

    #
    # Search settings
    #

    # Whether to quantize the vector for faster search the full scan
    quantize_scan: bool = field(
        default=True,
        metadata={
            "help": "Quantize vectors for faster full collection scans",
            "cli_name": "quantize-scan",
        },
    )
    # Load quantized vectors in memory for faster search
    quantize_preload: bool = field(
        default=False,
        metadata={
            "help": "Preload quantized vectors in memory",
            "cli_name": "quantize-preload",
        },
    )

    # Weights for combining FTS and vector search results
    weight_fts: float = field(
        default=1.0,
        metadata={"help": "Weight applied to full text search scores"},
    )
    weight_vec: float = field(
        default=1.5,
        metadata={"help": "Weight applied to vector similarity scores"},
    )

    #
    # Prompt templates
    # Some models are trained to work better with specific prompts
    # depending on the task. For example, Gemma models work better
    # when the prompt includes a task description.
    # More: https://huggingface.co/unsloth/embeddinggemma-300m-GGUF#prompt-instructions
    #

    use_prompt_templates: bool = field(
        default=True,
        metadata={
            "help": "Use the default prompt templates for embeddings",
            "cli_name": "prompt-templates",
        },
    )

    # Template to index documents for retrieval, use `{title}` with the title or the string `"none"`
    prompt_template_retrieval_document: str = field(
        default="title: {title} | text: {content}",
        metadata={"help": "Template applied to documents before indexing"},
    )
    prompt_template_retrieval_query: str = field(
        default='title: "none" | text: {content}',
        metadata={"help": "Template applied to the query prior to retrieval"},
    )

    #
    # Index settings
    #

    # Maximum size of a document to process (in bytes)
    max_document_size_bytes: int = field(
        default=5 * 1024 * 1024,
        metadata={"help": "Maximum size (in bytes) of a document before truncation"},
    )  # 5 MB
    # Zero means no limit
    max_chunks_per_document: int = field(
        default=1000,
        metadata={
            "help": "Maximum number of chunks generated per document (0 = unlimited)"
        },
    )
    # Number of top sentences to return per document
    top_k_sentences: int = field(
        default=5,
        metadata={"help": "Top sentences per document returned in retrieval results"},
    )

    #
    # Text generation
    #

    # model_path_text_gen: str = field(
    #     default="./models/unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q8_0.gguf",
    #     metadata={"help": "Path to the text generation model file (.gguf)"},
    # )
    model_path_text_gen: str = field(
        default="./models/unsloth/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q8_0.gguf",
        metadata={"help": "Path to the text generation model file (.gguf)"},
    )

    # Model parameters
    temp: float = field(
        default=1.0,
        metadata={"help": "Temperature for text generation"},
    )
    top_k: int = field(
        default=64,
        metadata={"help": "Top-K sampling value for text generation"},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-P (nucleus) sampling value for generation"},
    )
    top_p_min_keep: int = field(
        default=1,
        metadata={"help": "Minimum tokens kept when applying top-p sampling"},
    )
    min_p: float = field(
        default=0.0,
        metadata={"help": "Minimum probability threshold for min-p sampling"},
    )
    min_p_min_keep: int = field(
        default=1,
        metadata={"help": "Minimum tokens kept when applying min-p sampling"},
    )
    penaltiy_n_tokens: int = field(
        default=1024,
        metadata={"help": "Number of tokens considered for repetition penalties"},
    )
    penalty_repeat: float = field(
        default=1.1,
        metadata={"help": "Repeat penalty value"},
    )
    penalty_frequency: float = field(
        default=0.0,
        metadata={"help": "Frequency penalty applied during generation"},
    )
    penalty_presence: float = field(
        default=0.0,
        metadata={"help": "Presence penalty applied during generation"},
    )
    random_seed: int = field(
        default=-1,
        metadata={"help": "Random seed used for generation (-1 for random)"},
    )  # -1 means random seed

    # See: https://github.com/sqliteai/sqlite-ai/blob/main/API.md#llm_model_loadpath-text-options-text
    other_model_options_text_gen: str = field(
        default="",
        metadata={"help": "Additional options for the text generation model loader"},
    )
    # See: https://github.com/sqliteai/sqlite-ai/blob/main/API.md#llm_context_createoptions-text
    other_context_options_text_gen: str = field(
        default="",
        metadata={"help": "Extra context creation parameters for text generation"},
    )

    # Max context size to feed the model for generation
    context_size_text_gen: int = field(
        default=32000,
        metadata={"help": "Maximum context window passed to the generation model"},
    )
    # Max input tokens to the model for generation
    max_tokens: int = field(
        default=32000,
        metadata={"help": "Maximum tokens fed to the generation model"},
    )

    n_predict: int = field(
        default=800,
        metadata={"help": "Maximum number of tokens to predict in one call"},
    )

    # Answers is generated from retrieved documents
    # with RRF combined score below this threshold (0.0 best)
    results_threshold: float = field(
        default=0.020,
        metadata={"help": "Minimum combined RRF score required for generated answers"},
    )

    def get_context_options_embedding(self) -> str:
        """Get the context options for embeddings generation."""
        options = {
            "n_ctx": self.chunk_size,
            "embedding_type": self.vector_type,
            "pooling_type": self.pooling_type,
            "generate_embedding": 1,
            "normalize_embedding": 1,
        }

        return ",".join(f"{k}={v}" for k, v in options.items()) + (
            f",{self.other_model_context_options}"
            if self.other_model_context_options
            else ""
        )

    def get_context_options_text_generation(self) -> str:
        """Get the context options for text generation."""
        options = {
            "n_ctx": self.context_size_text_gen,
            "context_size": self.context_size_text_gen,
            "max_tokens": self.max_tokens,
            "n_predict": self.n_predict,
        }

        return ",".join(f"{k}={v}" for k, v in options.items()) + (
            f",{self.other_context_options_text_gen}"
            if self.other_context_options_text_gen
            else ""
        )

    def get_vector_init_options(self) -> str:
        """Get the vector init options for the vector store."""
        options = {"type": self.vector_type, "dimension": self.embedding_dim}
        return ",".join(f"{k}={v}" for k, v in options.items()) + (
            f",{self.other_vector_options}" if self.other_vector_options else ""
        )


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
            or new_settings.pooling_type != current_settings.pooling_type
        )
