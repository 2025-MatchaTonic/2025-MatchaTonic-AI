# app/rag/ingest.py
import argparse
from pathlib import Path
from typing import Iterable

from app.rag.vectorstore import get_vectorstore

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".csv", ".json"}


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    normalized = text.strip()
    if not normalized:
        return []

    stride = max(1, chunk_size - chunk_overlap)
    chunks = []
    for start in range(0, len(normalized), stride):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(normalized):
            break
    return chunks


def run_ingestion(
    source_dir: str = "data",
    *,
    phase: str | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> int:
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"source directory not found: {root}")

    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise RuntimeError("vectorstore unavailable: check OPENAI/PINECONE env vars")

    texts: list[str] = []
    metadatas: list[dict] = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        raw = _read_text(path)
        chunks = _chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        relative = str(path.relative_to(root)).replace("\\", "/")

        for idx, chunk in enumerate(chunks):
            metadata = {
                "source": relative,
                "chunk_index": idx,
            }
            if phase:
                metadata["phase"] = phase

            texts.append(chunk)
            metadatas.append(metadata)

    if not texts:
        return 0

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    return len(texts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest local docs into Pinecone")
    parser.add_argument(
        "--source-dir",
        default="data",
        help="directory containing markdown/text docs",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="optional metadata.phase value to attach to all chunks",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    count = run_ingestion(
        source_dir=args.source_dir,
        phase=args.phase,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"ingested_chunks={count}")
