from attr import dataclass

from .document import Document


@dataclass
class DocumentResult:
    document: Document

    snippet: str

    combined_rank: float
    vec_rank: float | None = None
    fts_rank: float | None = None

    vec_distance: float | None = None
    fts_score: float | None = None
