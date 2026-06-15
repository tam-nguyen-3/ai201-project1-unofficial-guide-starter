"""Query the ChromaDB index for the top-k most relevant chunks.

Usage as a library:
    from src.retrieve import retrieve
    results = retrieve("What is the cost of living in Taipei?", k=5)

Usage as a script (runs 3 of the 5 evaluation-plan queries):
    .venv/bin/python -m src.retrieve
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.embed_store import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


@dataclass
class Retrieved:
    rank: int
    distance: float
    source: str
    text: str
    metadata: dict


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(query: str, k: int = 5) -> list[Retrieved]:
    """Embed `query` and return the top-k chunks (lower distance = more similar)."""
    model = _get_model()
    coll = _get_collection()

    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = coll.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        Retrieved(
            rank=i + 1,
            distance=float(d),
            source=m["source"],
            text=doc,
            metadata=dict(m),
        )
        for i, (doc, m, d) in enumerate(zip(docs, metas, dists))
    ]


# ---------------------------------------------------------------------------
# Test harness: run 3 of the 5 evaluation-plan queries
# ---------------------------------------------------------------------------

EVAL_QUERIES = [
    "What are some cultural shocks students can expect living in Taiwan?",
    "What are some must-try foods in Taiwan?",
    "What should I know before going to Taipei as a student?",
    "What activities do foreigners in Taiwan think all visitors should try?",
    "What is the estimated monthly living cost for a student in Taipei?",
]


def _print_results(query: str, results: list[Retrieved]) -> None:
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    for r in results:
        print(
            f"\n[Rank {r.rank}]  distance={r.distance:.3f}  "
            f"source={r.source}  pos={r.metadata['position']}  "
            f"author=u/{r.metadata['author']}"
        )
        print("-" * 78)
        print(r.text)
        print("-" * 78)
    print()


def main():
    # Run 3 of the 5 eval queries (one from each major topic)
    test_queries = [EVAL_QUERIES[0], EVAL_QUERIES[1], EVAL_QUERIES[4]]
    for q in test_queries:
        _print_results(q, retrieve(q, k=5))


if __name__ == "__main__":
    main()
