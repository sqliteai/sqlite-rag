import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from sqlite_rag.connection_registry import ConnectionRegistry


@dataclass
class LLMModel:
    """Manage the lifecycle of a model bound to a specific SQLite connection."""

    model_path: str
    model_options: str
    context_options: str
    registry: ConnectionRegistry
    connection_name: str
    _loaded: bool = field(default=False, init=False)

    def ensure_loaded(self) -> sqlite3.Connection:
        """Ensure the underlying model is loaded and return its connection."""
        conn = self.registry.get_connection(self.connection_name)
        if self._loaded:
            return conn

        path = Path(self.model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Model '{self.model_path}' file not found. "
                "Verify the configured path or download the model."
            )

        conn.execute(
            "SELECT llm_model_load(?, ?);",
            (self.model_path, self.model_options),
        )

        conn.execute(
            "SELECT llm_context_create(?);",
            (self.context_options,),
        )

        self._loaded = True
        return conn

    def unload(self) -> None:
        """Release the model from memory if it is currently loaded."""
        if not self._loaded:
            return

        conn = self.registry.get_connection(self.connection_name)
        try:
            conn.execute("SELECT llm_model_free();")
        except sqlite3.ProgrammingError:
            # Connection might already be closed by the caller.
            pass
        finally:
            self._loaded = False
