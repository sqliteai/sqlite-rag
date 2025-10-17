from dataclasses import dataclass, field

from .document import Document
from .sentence_result import SentenceResult


@dataclass
class DocumentResult:
    document: Document

    chunk_id: int
    snippet: str

    combined_rank: float
    vec_rank: float | None = None
    fts_rank: float | None = None

    vec_distance: float | None = None
    fts_score: float | None = None

    # highlight sentences
    sentences: list[SentenceResult] = field(default_factory=list)
