# The Unofficial Guide — Project 1

A grounded RAG over Reddit threads from r/taiwan and r/Taipei that answers questions about living and studying abroad in Taipei.

**Pipeline:** [src/ingest.py](src/ingest.py) loads + chunks → [src/embed_store.py](src/embed_store.py) embeds + persists to ChromaDB → [src/retrieve.py](src/retrieve.py) top-k cosine search → [src/generate.py](src/generate.py) grounded answer via Groq → [src/app.py](src/app.py) Gradio UI.

**To reproduce:**
```bash
pip install -r requirements.txt
python -m src.ingest          # chunk raw Reddit JSON → documents/chunks.jsonl
python -m src.embed_store     # embed + write ChromaDB index
python -m src.app             # launch Gradio at http://127.0.0.1:7860
```

---

## Domain

The living-and-studying-abroad experience in **Taipei, Taiwan**, targeted at exchange students. This knowledge is valuable because official school channels (study-abroad offices, brochures) only provide curated, surface-level information. Real insights — actual monthly budgets, cultural surprises, food recommendations, day-to-day logistics — come from people who have lived it. Reddit communities like r/taiwan and r/Taipei aggregate candid, first-hand accounts from expats, exchange students, and long-term residents, covering topics that official sources rarely address: realistic cost breakdowns in NTD, culture-shock moments, neighborhood tips, food culture, and social experiences.

---

## Document Sources

11 Reddit threads (saved as JSON in [documents/raw/](documents/raw/)) spanning cost-of-living, culture shock, food, and visitor-tips subtopics.

| # | Source | Type | URL |
|---|--------|------|-----|
| 1 | r/taiwan — *Students living expenses* | Reddit thread | https://www.reddit.com/r/taiwan/comments/1rsrvkp/students_living_expenses/ |
| 2 | r/Taipei — *Cost of living in Taipei* | Reddit thread | https://www.reddit.com/r/Taipei/comments/1l65xjy/cost_of_living_in_taipei/ |
| 3 | r/taiwan — *Living cost as a student* | Reddit thread | https://www.reddit.com/r/taiwan/comments/1mchc8q/living_cost_as_a_student/ |
| 4 | r/Taipei — *All the things I wish I knew before moving to Taipei* | Reddit thread | https://www.reddit.com/r/Taipei/comments/1qna074/all_the_things_i_wish_i_knew_before_moving_to/ |
| 5 | r/taiwan — *Stereotypes Taiwanese have about other people* | Reddit thread | https://www.reddit.com/r/taiwan/comments/1tidhl4/stereotypes_that_taiwanese_have_about_other/ |
| 6 | r/taiwan — *What are some culture shocks in Taiwan?* | Reddit thread | https://www.reddit.com/r/taiwan/comments/1fzjrzg/what_are_some_culture_shocks_in_taiwan/ |
| 7 | r/taiwan — *Must-try food or restaurant in Taipei* | Reddit thread | https://www.reddit.com/r/taiwan/comments/1l67an4/must_try_food_or_restaurant_in_taipei/ |
| 8 | r/Taipei — *Unique restaurants in Taipei* | Reddit thread | https://www.reddit.com/r/Taipei/comments/1nft36e/unique_restaurants_in_taipei/ |
| 9 | r/taiwan — *I am an exchange student in Taiwan. Here is what it's actually like* | Reddit thread | https://www.reddit.com/r/taiwan/comments/1s4faq3/i_am_an_exchange_student_in_taiwan_here_iswhat/ |
| 10 | r/taiwan — *Foreigners, what are you doing here?* | Reddit thread | https://www.reddit.com/r/taiwan/comments/1rv58kr/foreigners_what_are_you_doing_here/ |
| 11 | r/taiwan — *What thing do you do in Taiwan that you think all foreigners should try?* | Reddit thread | https://www.reddit.com/r/taiwan/comments/1as5dki/what_thing_do_you_do_in_taiwan_that_you_think_all/ |

---

## Chunking Strategy

Implementation: [src/ingest.py](src/ingest.py) — `parse_thread()`, `_walk_comments()`, `chunk_thread()`.

**Chunk size:** Hard-capped at **256 tokens** (the context window of `all-MiniLM-L6-v2`). Typical chunks are 60–150 tokens. One chunk per comment by default; comments longer than the budget are split at paragraph → sentence → raw-token boundaries.

**Overlap:** **None** between adjacent chunks. Continuity is preserved structurally via a header prepended to every chunk:
```
POST TITLE: <thread title>
REPLY TO u/<parent author>: <first ~220 chars of parent body>...

<comment body>
```
This is functionally equivalent to overlap but semantically richer for Reddit's reply tree: a sibling comment is no more contextually relevant than a random other comment, so blindly repeating the previous chunk's tail would just add noise.

**Why these choices fit your documents:** Reddit threads are reply trees, not linear documents. A comment's meaning often depends on its parent ("I disagree, the real answer is X"). The structure-aware header makes every chunk standalone for retrieval. Capping at 256 tokens prevents a long rant from bundling multiple distinct opinions into one embedding (which dilutes retrieval precision).

**Preprocessing:** HTML entity decoding (`&amp;` → `&`), markdown link flattening (`[text](url)` → `text`), whitespace normalization. `[deleted]`, `[removed]`, empty bodies, and Redact-removal placeholders (*"This post no longer contains its original content..."*) are dropped; their replies are re-parented upward so the tree stays intact.

**Final chunk count:** **1,217** chunks across 11 threads (median 97 chunks/thread, range 14–228). Token distribution: min 28, median 93, max 256.

---

## Embedding Model

Implementation: [src/embed_store.py](src/embed_store.py), [src/retrieve.py](src/retrieve.py).

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`. 384-dim vectors, runs locally on CPU, no API key, no rate limits, ~10 s to embed all 1,217 chunks. Embeddings are L2-normalized (`normalize_embeddings=True`) so the ChromaDB collection's `cosine` distance metric is well-behaved. Stored in a persistent ChromaDB collection (`taipei_guide`) at [chroma_db/](chroma_db/).

**Production tradeoff reflection:** If cost weren't a constraint, I'd revisit four axes:

- **Context length** — MiniLM's 256-token cap forces me to split longer comments, occasionally cutting a long rant mid-thought. A model like `bge-large-en-v1.5` (512 tokens) or `text-embedding-3-large` (8 k tokens) would let me embed a long comment as a single chunk and keep more local context.
- **Multilingual support** — MiniLM is English-only. Many retrieved chunks contain Mandarin restaurant names (真善美牛肉麵, 麻醬面, 山胡椒) and the model treats those as low-signal tokens. A model like `multilingual-e5-large` would embed those names meaningfully and surface posts that mix English and Mandarin.
- **Domain-specific vocabulary** — Terms like *MRT*, *EasyCard*, *NTD*, *YouBike*, *scooter culture* are Taipei-specific. A general-purpose embedding model handles them weakly. Fine-tuning on Reddit r/taiwan posts (or using a model with broader colloquial coverage like OpenAI's `text-embedding-3-large`) would tighten retrieval on these terms.
- **Latency vs. quality** — MiniLM is ~80 MB and runs at ~140 chunks/s on CPU; large hosted models run in 100–300 ms per query. For a Gradio demo with one user at a time the larger model is fine; for many concurrent users the local MiniLM stays cheaper.

---

## Grounded Generation

Implementation: [src/generate.py](src/generate.py). LLM: Groq's `llama-3.3-70b-versatile`, temperature 0.2.

**System prompt grounding instruction** ([src/generate.py:25](src/generate.py#L25)):

```
You answer questions about living and studying abroad in Taipei, Taiwan,
using candid Reddit discussions as your only source of truth.

STRICT RULES — follow all of them:

1. Answer ONLY using the numbered context excerpts the user provides.
   Do NOT use general knowledge, training data, or assumptions beyond
   what is in the excerpts.

2. Cite every fact inline using the format (Source: <URL>) immediately
   after the statement, where <URL> is the Reddit link from the excerpt
   header. If multiple excerpts support a claim, cite them all.

3. If the context does not contain information to answer the question,
   respond with EXACTLY this sentence and nothing else:
   "I don't have enough information from the sources to answer this."
   Do not try to be helpful by guessing or supplementing from outside
   knowledge.

4. Quote specific numbers, currency amounts (e.g., NTD), place names,
   and food names exactly as they appear in the excerpts.

5. Be concise. 2–4 sentences for short questions, up to ~150 words for
   broader ones. Do not invent connecting facts to "smooth out" the answer.
```

**How source attribution is surfaced in the response:** Two complementary layers (belt + suspenders):

1. **Inline LLM citations** — every claim ends with `(Source: <Reddit URL>)`. The URL is taken from a per-chunk metadata field populated at index-build time from a hard-coded `SOURCE_URL_MAP` derived from `REDDIT_THREAD_URLS` in [src/ingest.py](src/ingest.py).
2. **Programmatic source block** — after generation, [`_format_sources_block()`](src/generate.py#L77) appends a deduplicated "Sources consulted" list (URL — thread title) so the user gets a clean attribution footer even if the LLM forgets a citation. Suppressed when the model emits the refusal sentence so it doesn't falsely claim to have used sources.

**Refusal mechanism:** The system prompt mandates an exact refusal sentence for out-of-domain questions. End-to-end test with "What is the average tuition fee at the University of Tokyo?" returns the refusal verbatim with no programmatic source block appended.

---

## Evaluation Report

All 5 eval-plan queries from [planning.md](planning.md) run end-to-end through retrieval (k=5) + generation. Distance scores in the table are the top-1 retrieved chunk's cosine distance (lower = better; <0.4 is a strong match for this model).

| # | Question | Expected answer | System response (summarized) | Top dist | Retrieval | Accuracy |
|---|----------|-----------------|------------------------------|----------|-----------|----------|
| 1 | What are some cultural shocks students can expect living in Taiwan? | Garbage-truck lining-up, friendly people, mold/dust, scooter traffic | 7-11 everywhere, people mind their own business, no friend-introductions, fluorescent lighting + exposed wires, long work hours, lack of clinic privacy | 0.184 | Relevant | Partially accurate — on-topic but covers different items than expected (the thread itself emphasized different shocks) |
| 2 | What are some must-try foods in Taiwan? | Beef noodle soup, xiaolongbao, gua bao, popcorn chicken, shaved ice, stinky tofu | 真善美牛肉麵 (beef noodles), danbing, Taiwanese breakfast (soy milk, fantuan, youtiao), night market food (black pepper bun, oyster omelettes), stinky tofu, boba, pineapple cakes, shave ice | 0.198 | Relevant | Accurate — overlaps on beef noodles, shaved ice, stinky tofu; adds danbing and pineapple cakes; missed xiaolongbao and gua bao because they don't appear in retrieved chunks |
| 3 | What should I know before going to Taipei as a student? | EasyCard for MRT, hot humid weather, night markets, Jiufen / Elephant Mountain / Longshan Temple | Learn Chinese, memorize practical characters, ~250 NTD per meal at nicer restaurants, make local English-speaking friends | 0.269 | Relevant | Partially accurate — retrieved from the *right* thread ("Things I wish I knew before moving") but that thread emphasizes language/friends over the tourist-prep items in the expected answer |
| 4 | What activities do foreigners think all visitors should try? | Hiking, YouBike, MRT, night markets, day trips | Taking the bus, drinking in parks, hiking mountains | 0.221 | Relevant | Partially accurate — top-5 only surfaced 3 activities because the source thread has dozens of one-line replies and many activities sit at ranks 6–20 |
| 5 | What is the estimated monthly living cost for a student in Taipei? | 15,000–20,000 NTD budget; 25,000–30,000 NTD comfortable | "At least $30,000 NTD" (~$1,000 USD) for a comfortable life; 20,000 NTD/month for a higher standard | 0.186 | Relevant | Accurate — figures match the expected range and are correctly attributed |

**Retrieval quality:** **Relevant** across all 5 queries — every top-1 result came from the most topically appropriate thread, with distances 0.184–0.269 (well below the 0.6–0.7 weak-match threshold).
**Response accuracy:** **Partially accurate overall** — 2 of 5 fully accurate (Q2, Q5), 3 of 5 partially accurate. The partial-accuracy cases are not grounding failures: the model faithfully reproduces what's in the retrieved chunks. They're a mismatch between what planning.md's *expected answer* assumes the Reddit threads contain and what the threads actually emphasize.

---

## Failure Case Analysis

**Question that failed:** *"What activities do foreigners in Taiwan think all visitors should try?"* (Q4)

**What the system returned:** A short list of three activities — taking the bus, drinking in parks, hiking mountains. The expected answer named five (hiking, YouBike, MRT, night markets, day trips outside Taipei) and the source thread in fact contains dozens of one-line suggestions covering all of these and more.

**Root cause (tied to a specific pipeline stage):** **Retrieval k is too small for list-style questions over a thread of one-liner replies.** The source thread ("What thing do you do in Taiwan that you think all foreigners should try?") consists almost entirely of short, single-activity top-level comments — each chunked into its own small embedding. With k=5, retrieval surfaces only ~5 of the ~228 chunks from that thread, and which ones land on top is largely driven by surface-token overlap with the query rather than meaningful semantic depth. Activities like *YouBike*, *MRT*, and *night markets* live in chunks that rank 6th, 11th, 17th, etc., and never reach the LLM.

This is the inverse of the more common chunking pitfall: chunks are not *too small* in tokens (they're well-formed), they're too small in **coverage**. A list query needs many shallow chunks; a "explain this concept" query needs few deep ones. A fixed k=5 cannot serve both.

**What you would change to fix it:** Two reasonable options:
1. **Adaptive k by query type** — detect list-shaped queries ("What...", "List...", plural noun + "should try") and bump k to 15–20, with light reranking to surface diverse activities.
2. **Coarser chunking for one-liner-heavy threads** — if a thread's median comment length is <50 tokens, group siblings into a single chunk so the embedding captures "this thread says X, Y, Z" rather than three separate "X" embeddings competing on cosine distance.

Option 2 is more principled; option 1 is faster to ship and gives most of the benefit. I'd start with option 1 and revisit option 2 if list-query failures persist.

---

## Spec Reflection

**One way the spec helped you during implementation:** The Chunking Strategy section in [planning.md](planning.md) made the *no-overlap + metadata-prepending* decision explicit before I wrote any code. That single design choice — "prepend post title + parent comment as a header to make each chunk self-contained" — is what makes retrieval over Reddit reply trees actually work. Without the spec, the path of least resistance would have been generic 500-token chunks with 50-token overlap, which would happily mix sibling comments from unrelated subthreads and produce noisy embeddings. The spec forced me to think about Reddit's structure first and pick a chunking approach that matched it.

**One way your implementation diverged from the spec, and why:** Two divergences, both small:

1. **Source attribution format.** The spec called for storing chunks in ChromaDB with metadata for attribution but didn't specify the citation format. I initially used the saved JSON filename (`cost_of_living_in_taipei.json`) as the `source` field — fine for debugging, but useless as a citation. I switched to the canonical Reddit URL (via `SOURCE_URL_MAP` in [src/embed_store.py](src/embed_store.py)) so inline citations are clickable links a reader can verify.
2. **Added a Redact-removal filter** that the planning.md "Anticipated Challenges" section did not predict. Some Reddit users delete their accounts via the Redact tool, which replaces every comment body with a generic disclaimer + a string of random words ("knee hungry grandfather unpack dime middle grey..."). These chunks were semantic poison — high random-token entropy that could trick cosine similarity into spurious matches. Filtering them at ingestion time (alongside `[deleted]` and `[removed]`) was a one-line addition that protected retrieval quality.

---

## AI Usage

**Instance 1 — Building the ingestion + chunking pipeline**

- *What I gave the AI:* The Chunking Strategy and Architecture sections of [planning.md](planning.md), plus the instruction "load Reddit links as JSON, preprocess them, separate into chunks matching the spec, and print 5 representative chunks for inspection." I also gave the rubric on what makes a good vs. bad chunk.
- *What it produced:* [src/ingest.py](src/ingest.py) with `fetch_thread()` (using `requests` against `<url>.json`), `parse_thread()` (walks the reply tree, cleans text), `chunk_thread()` (one chunk per comment, prepends title + parent excerpt, splits long bodies paragraph → sentence → token).
- *What I changed or overrode:* When the `requests`-based fetcher hit Reddit's anti-scraping 403 (Reddit now requires OAuth for nearly all programmatic access in 2026), I rejected the AI's first fallback suggestion (Pushshift / PullPush mirrors — they don't have these specific threads) and directed it to switch to a disk-loader strategy. I manually saved each thread's JSON from the browser (which carries cookies) into [documents/raw/](documents/raw/) under the slug filename, and the AI rewrote the loader to read by slug. This kept the rest of the pipeline unchanged.

**Instance 2 — Adding the Redact filter**

- *What I gave the AI:* A specific bad chunk from the output ("REPLY TO u/Asian_Quokka_: ... *This post no longer contains its original content...* knee hungry grandfather unpack dime middle grey rain fanatical telephone"), with the request to filter all such chunks at ingestion time.
- *What it produced:* A two-line change: a `_REDACT_MARKER` constant matching the unique disclaimer prefix, and an `_is_redacted()` helper called alongside the existing `[deleted]`/`[removed]` checks in [`_walk_comments()`](src/ingest.py#L142). Children of redacted comments get re-parented upward so the reply tree stays intact.
- *What I changed or overrode:* I scoped the request narrowly — drop the whole comment, do not try to salvage anything from the body. The chunk count dropped by exactly one (1,218 → 1,217), confirming the filter was conservative. Verified no surviving chunks contain the marker string with a one-line grep.
