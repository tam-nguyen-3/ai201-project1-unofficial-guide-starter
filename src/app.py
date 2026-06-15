"""Gradio interface for the Taipei Unofficial Guide RAG.

Run with:
    .venv/bin/python -m src.app
Then open http://127.0.0.1:7860 in a browser.
"""

from __future__ import annotations

import gradio as gr

from src.generate import generate_answer

EXAMPLE_QUERIES = [
    "What are some must-try foods in Taiwan?",
    "What is the estimated monthly living cost for a student in Taipei?",
    "What are some cultural shocks students can expect living in Taiwan?",
    "What activities do foreigners in Taiwan recommend all visitors try?",
    "What should I know before going to Taipei as a student?",
]


def answer_fn(query: str, k: int) -> tuple[str, str]:
    """Return (answer markdown, retrieved-chunks markdown)."""
    if not query or not query.strip():
        return "_Please enter a question._", ""

    result = generate_answer(query.strip(), k=int(k))

    debug_blocks = []
    for r in result.sources:
        debug_blocks.append(
            f"**[Rank {r.rank}]** dist=`{r.distance:.3f}`  "
            f"source=`{r.source}`  u/{r.metadata['author']}\n\n"
            f"> {r.text.replace(chr(10), chr(10) + '> ')}"
        )
    debug_md = "\n\n---\n\n".join(debug_blocks) if debug_blocks else "_no chunks_"
    return result.answer, debug_md


with gr.Blocks(title="Taipei Unofficial Guide") as demo:
    gr.Markdown("# Taipei Unofficial Guide")
    gr.Markdown(
        "Grounded RAG over Reddit threads from r/taiwan and r/Taipei. "
        "Answers are drawn **only** from retrieved excerpts; the system will "
        "refuse to answer questions the documents don't cover."
    )

    with gr.Row():
        query = gr.Textbox(
            label="Your question",
            placeholder="e.g. What should I budget per month as an exchange student?",
            lines=2,
            scale=4,
        )
        k = gr.Slider(
            minimum=1, maximum=10, step=1, value=5,
            label="Top-k chunks", scale=1,
        )
    submit = gr.Button("Ask", variant="primary")

    answer = gr.Markdown(label="Answer", value="_Ask a question to see an answer here._")
    with gr.Accordion("Retrieved chunks (for debugging grounding)", open=False):
        chunks = gr.Markdown(value="_Submit a query to see retrieved chunks._")

    submit.click(answer_fn, inputs=[query, k], outputs=[answer, chunks])
    query.submit(answer_fn, inputs=[query, k], outputs=[answer, chunks])

    gr.Examples(
        examples=[[q] for q in EXAMPLE_QUERIES],
        inputs=[query],
        label="Example queries",
    )


if __name__ == "__main__":
    demo.launch()
