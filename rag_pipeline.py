from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import config
import openrouter_client
import pdf_ingest
from vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are Ling-3.0-Flash-Fin (inclusionAI), a financial analyst assistant "
    "reviewing excerpts pulled from a company's financial filing (text, tables "
    "rendered as markdown, and descriptions of figures/charts). Answer "
    "strictly from the excerpts given - do not use outside knowledge of the "
    "company, even if you recognize it, since the excerpts are the only source "
    "you can verify. This restriction applies only to facts about the company "
    "and its filing. You may freely answer questions about yourself (who you "
    "are, what model you are, what you can do) and general conversational "
    "questions.\n\n"
    "Accuracy rules:\n"
    "- Reconcile numbers across excerpts when more than one touches the "
    "question (e.g. a table and a text passage); if they conflict, point "
    "that out rather than picking one silently.\n"
    "- Every figure or claim must be traceable: cite the page(s) it came "
    "from, e.g. '(page 3)'.\n"
    "- If the excerpts don't contain the answer, say so plainly instead of "
    "guessing or filling in from memory.\n"
    "- This is not investment advice; note that any valuation or forward-"
    "looking conclusion needs professional review.\n\n"
    "Output format:\n"
    "- Do not show your step-by-step reasoning, only the conclusion.\n"
    "- Start with the direct answer in one or two sentences, then add "
    "supporting detail only if needed.\n"
    "- Use short bullet points for supporting figures, one fact per bullet, "
    "each ending with its page citation.\n"
    "- Use a markdown table when comparing multiple periods, segments, or "
    "line items; put the period or unit in the column header.\n"
    "- State every figure with its currency, unit (thousands/millions), and "
    "period, and format numbers consistently (e.g. $1,234.5 million, 12.3%).\n"
    "- Plain markdown only: no LaTeX, no headers for short answers, and bold "
    "only for the key figure or the direct answer.\n"
    "- Keep the whole response concise and directly usable; place any "
    "conflict or missing-data note on its own line at the end."
)


class RAGPipeline:
    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self.store = VectorStore()
        if config.INDEX_PATH.exists():
            self.store.load(config.INDEX_PATH)

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        return self._embedder

    def ingest(
        self,
        pdf_path: Path,
        api_key: str,
        vision_model: str,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> dict:
        docs = pdf_ingest.ingest_pdf(pdf_path, api_key, vision_model, progress_cb)
        if not docs:
            raise ValueError("No text, tables, or figures could be extracted from this PDF.")

        if progress_cb:
            progress_cb(f"Embedding {len(docs)} chunks")
        contents = [d["content"] for d in docs]
        embeddings = self.embedder.encode(
            contents, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        self.store.build(np.asarray(embeddings, dtype=np.float32), docs)
        self.store.save(config.INDEX_PATH)

        counts = {"text": 0, "table": 0, "figure": 0}
        for d in docs:
            counts[d["type"]] = counts.get(d["type"], 0) + 1
        return counts

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        if not self.store.is_ready():
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        return self.store.hybrid_search(q_emb, query, top_k=top_k)

    def answer(self, query: str, api_key: str, text_model: str, top_k: int = config.TOP_K):
        if not self.store.is_ready():
            return (
                "No document has been indexed yet. Put a PDF in the input/ folder "
                "and reload the page.",
                [],
            )

        results = self.retrieve(query, top_k=top_k)
        context_blocks = []
        for r in results:
            tag = f"[{r['type'].upper()} | page {r['page']}]"
            context_blocks.append(f"{tag}\n{r['content']}")
        context = "\n\n---\n\n".join(context_blocks)

        user_prompt = (
            f"Context excerpts from the PDF:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        answer_text = openrouter_client.chat_completion(
            messages,
            model=text_model,
            api_key=api_key,
            temperature=config.TEXT_TEMPERATURE,
            top_p=config.TEXT_TOP_P,
            top_k=config.TEXT_TOP_K,
            max_tokens=config.TEXT_MAX_TOKENS,
        )
        return answer_text, results
