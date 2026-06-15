"""Grounded answer generation via Groq's llama-3.3-70b-versatile + retrieved chunks.

The grounding contract has three layers:
  1. System prompt: strict rules to answer ONLY from provided context, with a fixed
     refusal sentence when context is insufficient.
  2. Context formatting: each chunk is labeled with its source filename, thread
     title, and author so the model can cite it inline.
  3. Programmatic source list: after generation, a deduplicated "Sources consulted"
     block is appended (belt-and-suspenders attribution).

Usage:
    from src.generate import generate_answer
    result = generate_answer("How much should a student budget per month?", k=5)
    print(result.answer)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.retrieve import Retrieved, retrieve

GROQ_MODEL = "llama-3.3-70b-versatile"
REFUSAL_SENTENCE = (
    "I don't have enough information from the sources to answer this."
)

SYSTEM_PROMPT = f"""You answer questions about living and studying abroad in Taipei, Taiwan, using candid Reddit discussions as your only source of truth.

STRICT RULES — follow all of them:

1. Answer ONLY using the numbered context excerpts the user provides. Do NOT use general knowledge, training data, or assumptions beyond what is in the excerpts.

2. Cite every fact inline using the format (Source: <filename>) immediately after the statement. If multiple excerpts support a claim, cite them all: (Sources: a.json, b.json).

3. If the context does not contain information to answer the question, respond with EXACTLY this sentence and nothing else:
{REFUSAL_SENTENCE}
Do not try to be helpful by guessing or supplementing from outside knowledge. Do not say "based on the context" -- just emit the refusal sentence verbatim.

4. Quote specific numbers, currency amounts (e.g., NTD), place names, and food names exactly as they appear in the excerpts.

5. Be concise. 2-4 sentences for short questions, up to ~150 words for broader ones. Do not invent connecting facts to "smooth out" the answer.

Output format:
ANSWER: <your answer with inline (Source: <filename>) citations>
"""

USER_TEMPLATE = """Question: {query}

Context excerpts:
{context}

Answer the question using only the context excerpts above."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_context(results: list[Retrieved]) -> str:
    blocks = []
    for r in results:
        title = r.metadata.get("thread_title", "")
        author = r.metadata.get("author", "unknown")
        header = (
            f"[{r.rank}] Source: {r.source} | Thread: \"{title}\" | u/{author}"
        )
        blocks.append(f"{header}\n{r.text}")
    return "\n\n".join(blocks)


def _format_sources_block(results: list[Retrieved]) -> str:
    """Deduplicated 'Sources consulted' list, preserving first-seen order."""
    seen: dict[str, str] = {}
    for r in results:
        if r.source not in seen:
            seen[r.source] = r.metadata.get("thread_title", "")
    lines = [f"- {src} -- \"{title}\"" for src, title in seen.items()]
    return "Sources consulted:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    answer: str  # final response shown to the user (LLM output + source block)
    raw_answer: str  # raw LLM output, no appended sources
    sources: list[Retrieved]
    refused: bool


_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env (see .env.example)."
            )
        _CLIENT = Groq(api_key=api_key)
    return _CLIENT


def generate_answer(query: str, k: int = 5) -> GenerationResult:
    client = _get_client()
    retrieved = retrieve(query, k=k)
    user_msg = USER_TEMPLATE.format(
        query=query, context=_format_context(retrieved)
    )

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()

    refused = REFUSAL_SENTENCE in raw
    if refused:
        final = raw
    else:
        final = f"{raw}\n\n---\n{_format_sources_block(retrieved)}"

    return GenerationResult(
        answer=final, raw_answer=raw, sources=retrieved, refused=refused
    )


# ---------------------------------------------------------------------------
# End-to-end grounding test: 3 in-domain + 1 out-of-domain
# ---------------------------------------------------------------------------

TEST_QUERIES: list[tuple[str, str]] = [
    ("in-domain", "What are some must-try foods in Taiwan?"),
    ("in-domain", "What is the estimated monthly living cost for a student in Taipei?"),
    ("in-domain", "What are some cultural shocks students can expect living in Taiwan?"),
    ("out-of-domain", "What is the average tuition fee at the University of Tokyo?"),
]


def main():
    for label, q in TEST_QUERIES:
        print("=" * 80)
        print(f"[{label.upper()}] QUERY: {q}")
        print("=" * 80)
        result = generate_answer(q, k=5)
        print(result.answer)
        print()


if __name__ == "__main__":
    main()
