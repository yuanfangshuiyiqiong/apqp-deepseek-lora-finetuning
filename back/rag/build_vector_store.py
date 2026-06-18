import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List

from .embeddings import HashingEmbeddingModel
from .vector_store import chunk_text


SUPPORTED_SUFFIXES = {".txt", ".md", ".json"}


def build_vector_store(
    input_dir: str | Path,
    output_path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedding_dimension: int = 384,
) -> int:
    source_dir = Path(input_dir)
    target_path = Path(output_path)
    embedder = HashingEmbeddingModel(dimension=embedding_dimension)

    files = sorted(
        path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with target_path.open("w", encoding="utf-8") as handle:
        for doc_index, path in enumerate(files, start=1):
            text = _read_document(path)
            for chunk_index, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap), start=1):
                record = {
                    "id": f"doc_{doc_index:03d}_chunk_{chunk_index:04d}",
                    "text": chunk,
                    "embedding": embedder.embed(chunk),
                    "metadata": {
                        "source": path.name,
                        "path": str(path),
                        "chunk_index": chunk_index,
                        "category": "APQP",
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(_iter_json_text(payload))
    return path.read_text(encoding="utf-8")


def _iter_json_text(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_text(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_json_text(item)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local JSONL vector store from cleaned knowledge docs.")
    parser.add_argument("--input-dir", default="knowledge_docs")
    parser.add_argument("--output", default="knowledge_base/vector_store.jsonl")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--dimension", type=int, default=384)
    args = parser.parse_args(argv)

    count = build_vector_store(
        input_dir=args.input_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_dimension=args.dimension,
    )
    print(f"Built {count} vector records at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
