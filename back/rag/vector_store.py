import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class VectorRecord:
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    record: VectorRecord
    score: float


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start += step
    return chunks


class JsonlVectorStore:
    def __init__(self, records: Iterable[VectorRecord]):
        self.records = list(records)

    @classmethod
    def load(cls, path: str | Path) -> "JsonlVectorStore":
        store_path = Path(path)
        records: List[VectorRecord] = []
        if not store_path.exists():
            return cls(records)

        with store_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                try:
                    record = VectorRecord(
                        id=str(payload["id"]),
                        text=str(payload["text"]),
                        embedding=[float(value) for value in payload["embedding"]],
                        metadata=dict(payload.get("metadata") or {}),
                    )
                except KeyError as exc:
                    raise ValueError(f"Missing required field {exc} in {store_path}:{line_number}") from exc
                records.append(record)
        return cls(records)

    def search(self, query_embedding: List[float], top_k: int = 4, min_score: float = 0.0) -> List[SearchResult]:
        if top_k <= 0:
            return []

        results = [
            SearchResult(record=record, score=_cosine_similarity(query_embedding, record.embedding))
            for record in self.records
        ]
        results = [result for result in results if result.score >= min_score]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
