import gradio as gr

import config
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

TYPE_EMOJI = {"text": "📄", "table": "📊", "figure": "🖼️"}


def format_sources(results: list[dict]) -> str:
    if not results:
        return "_No sources retrieved._"
    lines = []
    for r in results:
        emoji = TYPE_EMOJI.get(r["type"], "•")
        snippet = r["content"].replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        lines.append(f"**{emoji} {r['type']} - page {r['page']}** (similarity {r['dense_score']:.2f})\n\n{snippet}")
    return "\n\n---\n\n".join(lines)


def load_document(api_key, vision_model, progress=gr.Progress()):
    """Index the PDF in input/ on page load, unless an index already exists."""
    if pipeline.store.is_ready():
        return "Document ready. Ask a question below."
    pdfs = sorted(config.INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        return "No PDF found in the input/ folder."
    pdf_path = pdfs[0]
    key = api_key or config.OPENROUTER_API_KEY

    try:
        pipeline.ingest(pdf_path, key, vision_model, progress_cb=lambda msg: progress(0, desc=msg))
    except Exception as e:
        return f"Ingestion failed: {e}"
    return f"Indexed **{pdf_path.name}**. Ask a question below."


def ask(question, history, api_key, text_model):
    key = api_key or config.OPENROUTER_API_KEY
    if not question or not question.strip():
        return history, "", gr.update()
    try:
        answer, results = pipeline.answer(question, key, text_model, top_k=config.TOP_K)
    except Exception as e:
        answer, results = f"Error: {e}", []

    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return history, "", format_sources(results)


CSS = """
#sources-box {
    height: 450px;
    overflow-y: auto;
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 12px;
}
"""

with gr.Blocks(title="PDF RAG (OpenRouter free models)", css=CSS) as demo:
    gr.Markdown(
        "# PDF RAG - text, tables & figures\n"
        "The PDF in `input/` is indexed automatically, then ask questions about it. "
        "Retrieval runs locally (sentence-transformers); answers and figure captions come from "
        "free models on [OpenRouter](https://openrouter.ai/models?max_price=0)."
    )

    with gr.Accordion("Setup", open=True):
        with gr.Row():
            api_key_box = gr.Textbox(
                label="OpenRouter API key (leave blank to use .env)",
                type="password",
                value="",
            )
            text_model_dd = gr.Dropdown(
                label="Text (answering) model",
                choices=config.TEXT_MODEL_CHOICES,
                value=config.TEXT_MODEL,
                allow_custom_value=True,
            )
            vision_model_dd = gr.Dropdown(
                label="Vision (figure captioning) model",
                choices=config.VISION_MODEL_CHOICES,
                value=config.VISION_MODEL,
                allow_custom_value=True,
            )

    status_box = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=450)
            question_box = gr.Textbox(label="Ask a question", placeholder="e.g. What was net revenue in Q3?")
            ask_btn = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            sources_md = gr.Markdown(value="_Ask a question to see sources._", elem_id="sources-box")

    demo.load(load_document, inputs=[api_key_box, vision_model_dd], outputs=[status_box])
    ask_btn.click(
        ask,
        inputs=[question_box, chatbot, api_key_box, text_model_dd],
        outputs=[chatbot, question_box, sources_md],
    )
    question_box.submit(
        ask,
        inputs=[question_box, chatbot, api_key_box, text_model_dd],
        outputs=[chatbot, question_box, sources_md],
    )

if __name__ == "__main__":
    demo.launch()
