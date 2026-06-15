"""Load Reddit threads, preprocess into a comment tree, and chunk for retrieval.

Pipeline (per planning.md):
  1. fetch_thread(url)            -> raw Reddit JSON (cached to documents/raw/)
  2. parse_thread(raw)            -> Thread with cleaned post + comment tree
  3. chunk_thread(thread, tok)    -> list[Chunk], one chunk per comment, with
                                    post title + parent comment prepended,
                                    hard-capped at 256 tokens (no overlap).
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from transformers import AutoTokenizer

REDDIT_THREAD_URLS = [
    "https://www.reddit.com/r/taiwan/comments/1rsrvkp/students_living_expenses/",
    "https://www.reddit.com/r/Taipei/comments/1l65xjy/cost_of_living_in_taipei/",
    "https://www.reddit.com/r/taiwan/comments/1mchc8q/living_cost_as_a_student/",
    "https://www.reddit.com/r/Taipei/comments/1qna074/all_the_things_i_wish_i_knew_before_moving_to/",
    "https://www.reddit.com/r/taiwan/comments/1tidhl4/stereotypes_that_taiwanese_have_about_other/",
    "https://www.reddit.com/r/taiwan/comments/1fzjrzg/what_are_some_culture_shocks_in_taiwan/",
    "https://www.reddit.com/r/taiwan/comments/1l67an4/must_try_food_or_restaurant_in_taipei/",
    "https://www.reddit.com/r/Taipei/comments/1nft36e/unique_restaurants_in_taipei/",
    "https://www.reddit.com/r/taiwan/comments/1s4faq3/i_am_an_exchange_student_in_taiwan_here_iswhat/",
    "https://www.reddit.com/r/taiwan/comments/1rv58kr/foreigners_what_are_you_doing_here/",
    "https://www.reddit.com/r/taiwan/comments/1as5dki/what_thing_do_you_do_in_taiwan_that_you_think_all/",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = REPO_ROOT / "documents"
RAW_DIR = DOCUMENTS_DIR / "raw"
CHUNKS_FILE = DOCUMENTS_DIR / "chunks.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Comment:
    id: str
    parent_id: str  # "post" for top-level, else the parent comment id
    author: str
    body: str
    children: list["Comment"] = field(default_factory=list)


@dataclass
class Thread:
    id: str
    title: str
    selftext: str
    permalink: str
    comments: list[Comment]


@dataclass
class Chunk:
    chunk_id: str
    thread_id: str
    thread_title: str
    permalink: str
    parent_id: str
    comment_id: str
    author: str
    text: str
    token_count: int


# ---------------------------------------------------------------------------
# Load: read Reddit thread JSON dumps from disk.
#
# Reddit blocks unauthenticated programmatic access, so each thread was saved
# manually from the browser (visiting the URL with ".json" appended) into
# documents/raw/<slug>.json, where <slug> is the trailing slug in the thread
# URL (the part after the thread id).
# ---------------------------------------------------------------------------


def _slug_from_url(url: str) -> str:
    parts = url.rstrip("/").split("/")
    return parts[parts.index("comments") + 2]


def load_thread(url: str, cache_dir: Path = RAW_DIR) -> list:
    """Load a Reddit thread JSON dump from disk by URL slug."""
    slug = _slug_from_url(url)
    path = cache_dir / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw thread file: {path}. "
            f"Save the JSON from {url}.json into this location."
        )
    return json.loads(path.read_text())


def load_all_threads(urls: list[str] = REDDIT_THREAD_URLS) -> list[list]:
    return [load_thread(u) for u in urls]


# ---------------------------------------------------------------------------
# Preprocess: clean text + flatten Reddit's reply tree
# ---------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://[^\s)]+|/[^\s)]+)\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_WS_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Decode HTML entities, drop markdown link syntax, normalize whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)
    lines = [_INLINE_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


_REDACT_MARKER = "This post no longer contains its original content."


def _is_redacted(body: str) -> bool:
    return _REDACT_MARKER in body


def _walk_comments(children: list[dict], parent_id: str) -> list[Comment]:
    """Recursively walk Reddit's nested comment listing, skipping deleted/empty
    comments and 'more' continuation tokens.  When a comment is dropped, its
    replies are re-parented to the dropped comment's parent so context stays
    intact."""
    out: list[Comment] = []
    for ch in children:
        if ch.get("kind") != "t1":
            continue
        d = ch["data"]
        body = (d.get("body") or "").strip()
        replies_field = d.get("replies") or {}
        sub_children = (
            replies_field.get("data", {}).get("children", [])
            if isinstance(replies_field, dict)
            else []
        )

        if not body or body in ("[deleted]", "[removed]") or _is_redacted(body):
            out.extend(_walk_comments(sub_children, parent_id))
            continue

        c = Comment(
            id=d["id"],
            parent_id=parent_id,
            author=d.get("author") or "unknown",
            body=clean_text(body),
            children=_walk_comments(sub_children, d["id"]),
        )
        out.append(c)
    return out


def parse_thread(raw: list) -> Thread:
    post_data = raw[0]["data"]["children"][0]["data"]
    comment_children = raw[1]["data"]["children"]
    return Thread(
        id=post_data["id"],
        title=clean_text(post_data.get("title", "")),
        selftext=clean_text(post_data.get("selftext", "")),
        permalink="https://www.reddit.com" + post_data.get("permalink", ""),
        comments=_walk_comments(comment_children, parent_id="post"),
    )


def iter_comments(comments: list[Comment]):
    for c in comments:
        yield c
        yield from iter_comments(c.children)


# ---------------------------------------------------------------------------
# Chunk: one chunk per comment, with post title + parent prepended.
# Hard-capped at MAX_TOKENS; over-long comments are split by paragraph then
# sentence, never overlapping.
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _ntokens(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def _excerpt(text: str, max_chars: int = 220) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _split_body(body: str, budget_tokens: int, tok) -> list[str]:
    """Split body into pieces each <= budget_tokens.  Paragraph-first, then
    sentence, then hard token cut as a last resort."""
    if budget_tokens < 16:
        budget_tokens = 16

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        paragraphs = [body.strip()]

    pieces: list[str] = []
    buf, buf_n = "", 0

    def flush():
        nonlocal buf, buf_n
        if buf.strip():
            pieces.append(buf.strip())
        buf, buf_n = "", 0

    def add_unit(text: str, sep: str):
        nonlocal buf, buf_n
        n = _ntokens(tok, text)
        if buf and buf_n + n > budget_tokens:
            flush()
        if buf:
            buf = buf + sep + text
            buf_n += n
        else:
            buf, buf_n = text, n

    for para in paragraphs:
        n = _ntokens(tok, para)
        if n <= budget_tokens:
            add_unit(para, "\n\n")
            continue
        # Paragraph too long -- flush and split into sentences.
        flush()
        for sent in _SENT_SPLIT_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            sn = _ntokens(tok, sent)
            if sn <= budget_tokens:
                add_unit(sent, " ")
                continue
            # Sentence still too long -- hard cut by tokens.
            flush()
            ids = tok.encode(sent, add_special_tokens=False)
            for i in range(0, len(ids), budget_tokens):
                pieces.append(tok.decode(ids[i:i + budget_tokens]).strip())
    flush()
    return pieces


def chunk_thread(thread: Thread, tok) -> list[Chunk]:
    by_id: dict[str, Comment] = {c.id: c for c in iter_comments(thread.comments)}

    # Treat the post body itself as a chunk-source if it has selftext.
    sources: list[Comment] = []
    if thread.selftext:
        sources.append(
            Comment(
                id=f"post_{thread.id}",
                parent_id="post",
                author="OP",
                body=thread.selftext,
            )
        )
    sources.extend(iter_comments(thread.comments))

    chunks: list[Chunk] = []
    for c in sources:
        if c.parent_id == "post":
            parent_label = "POST"
            parent_text = thread.selftext if c.id != f"post_{thread.id}" else ""
        else:
            parent = by_id.get(c.parent_id)
            parent_label = f"REPLY TO u/{parent.author}" if parent else "REPLY"
            parent_text = parent.body if parent else ""

        header_parts = [f"POST TITLE: {thread.title}"]
        if parent_text:
            header_parts.append(f"{parent_label}: {_excerpt(parent_text)}")
        header = "\n".join(header_parts) + "\n\n"
        header_n = _ntokens(tok, header)

        for i, piece in enumerate(_split_body(c.body, MAX_TOKENS - header_n, tok)):
            full = header + piece
            chunks.append(
                Chunk(
                    chunk_id=f"{thread.id}::{c.id}::{i}",
                    thread_id=thread.id,
                    thread_title=thread.title,
                    permalink=thread.permalink,
                    parent_id=c.parent_id,
                    comment_id=c.id,
                    author=c.author,
                    text=full,
                    token_count=_ntokens(tok, full),
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_chunks() -> list[Chunk]:
    raw = load_all_threads()
    threads = [parse_thread(r) for r in raw]
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    chunks: list[Chunk] = []
    for t in threads:
        chunks.extend(chunk_thread(t, tok))
    return chunks


def _save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def _representative_sample(chunks: list[Chunk], n: int) -> list[Chunk]:
    """Pick n chunks from distinct threads, each near the median length of its
    thread (avoids both 1-token-comment and outlier extremes)."""
    by_thread: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_thread.setdefault(c.thread_id, []).append(c)
    samples: list[Chunk] = []
    for tid, items in by_thread.items():
        items_sorted = sorted(items, key=lambda c: c.token_count)
        samples.append(items_sorted[len(items_sorted) // 2])
        if len(samples) >= n:
            break
    return samples[:n]


def main():
    print(f"Loading {len(REDDIT_THREAD_URLS)} Reddit threads "
          f"from {RAW_DIR.relative_to(REPO_ROOT)}/ ...")
    raw = load_all_threads()

    threads = [parse_thread(r) for r in raw]
    n_comments = sum(1 for t in threads for _ in iter_comments(t.comments))
    print(f"  parsed {len(threads)} threads, {n_comments} comments after cleaning")

    print(f"Loading tokenizer ({EMBED_MODEL})...")
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)

    print("Chunking...")
    chunks: list[Chunk] = []
    for t in threads:
        chunks.extend(chunk_thread(t, tok))

    token_counts = sorted(c.token_count for c in chunks)
    median = token_counts[len(token_counts) // 2]
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"  tokens/chunk -- min: {token_counts[0]}, "
          f"median: {median}, max: {token_counts[-1]}")
    print(f"  chunks per thread (min/median/max): "
          f"{_per_thread_stats(chunks)}")

    _save_chunks(chunks, CHUNKS_FILE)
    print(f"Saved {len(chunks)} chunks to {CHUNKS_FILE.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("5 REPRESENTATIVE CHUNKS")
    print("=" * 78)
    for i, c in enumerate(_representative_sample(chunks, 5), 1):
        print(f"\n--- Chunk {i}/{5} "
              f"[thread={c.thread_id}  comment={c.comment_id}  "
              f"author=u/{c.author}  tokens={c.token_count}] ---")
        print(c.text)
        print("-" * 78)


def _per_thread_stats(chunks: list[Chunk]) -> str:
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.thread_id] = counts.get(c.thread_id, 0) + 1
    vals = sorted(counts.values())
    return f"{vals[0]} / {vals[len(vals) // 2]} / {vals[-1]}"


if __name__ == "__main__":
    main()
