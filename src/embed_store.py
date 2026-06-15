"""Embed chunks from documents/chunks.jsonl and store them in ChromaDB.

Run once after ingestion to build the persistent vector index:
    .venv/bin/python -m src.embed_store
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingest import CHUNKS_FILE, REPO_ROOT

CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "taipei_guide"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingest` first."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _source_from_permalink(permalink: str) -> str:
    """Map a Reddit permalink back to the saved JSON filename in documents/raw/."""
    # https://www.reddit.com/r/taiwan/comments/<id>/<slug>/
    parts = permalink.rstrip("/").split("/")
    return f"{parts[-1]}.json"


def _build_metadatas(chunks: list[dict]) -> list[dict]:
    """Attach source filename and per-document position to each chunk."""
    position_in_doc: dict[str, int] = {}
    metas: list[dict] = []
    for c in chunks:
        pos = position_in_doc.get(c["thread_id"], 0)
        metas.append(
            {
                "source": _source_from_permalink(c["permalink"]),
                "position": pos,
                "thread_id": c["thread_id"],
                "thread_title": c["thread_title"],
                "comment_id": c["comment_id"],
                "author": c["author"],
                "parent_id": c["parent_id"],
                "permalink": c["permalink"],
                "token_count": c["token_count"],
            }
        )
        position_in_doc[c["thread_id"]] = pos + 1
    return metas


def build_index(persist_dir: Path = CHROMA_DIR) -> chromadb.Collection:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE.relative_to(REPO_ROOT)}")

    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # makes cosine distance well-behaved
    ).tolist()

    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to ChromaDB at {persist_dir.relative_to(REPO_ROOT)}/ ...")
    client = chromadb.PersistentClient(path=str(persist_dir))
    # Rebuild from scratch for reproducibility.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=_build_metadatas(chunks),
    )
    print(f"Stored {collection.count()} embeddings in collection '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    build_index()
