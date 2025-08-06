from enum import Enum
from typing import Callable, Optional


class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


class Logger:
    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        output_fn: Optional[Callable[[str], None]] = None,
    ):
        self.level = level
        self.output_fn = output_fn or print

    def debug(self, message: str):
        if self.level.value <= LogLevel.DEBUG.value:
            self.output_fn(f"DEBUG: {message}")

    def info(self, message: str):
        if self.level.value <= LogLevel.INFO.value:
            self.output_fn(f"{message}")

    def warning(self, message: str):
        if self.level.value <= LogLevel.WARNING.value:
            self.output_fn(f"WARNING: {message}")

    def error(self, message: str):
        if self.level.value <= LogLevel.ERROR.value:
            self.output_fn(f"ERROR: {message}")
