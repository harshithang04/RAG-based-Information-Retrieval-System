from pathlib import Path

import pymupdf

import config
import openrouter_client
import pdf_ingest


class TestFullPdf:
    def test_produces_text_and_table_chunks(self, docs):
        types = {d["type"] for d in docs}
        assert {"text", "table"} <= types

    def test_chunk_schema(self, docs):
        for d in docs:
            assert {"type", "page", "source", "content"} <= d.keys()
            assert d["type"] in {"text", "table", "figure"}
            assert d["content"].strip()
            assert d["source"] == "2022 Q3 AAPL.pdf"

    def test_pages_within_document(self, docs, pdf_path):
        n = len(pymupdf.open(pdf_path))
        assert n == 28
        assert all(1 <= d["page"] <= n for d in docs)

    def test_most_pages_have_content(self, docs):
        pages = {d["page"] for d in docs}
        assert len(pages) >= 24

    def test_income_statement_row_is_self_describing(self, docs):
        """A row chunk must carry the statement title and column headers, not just numbers."""
        tables = [d for d in docs if d["type"] == "table" and d["page"] == 4]
        chunk = next(t for t in tables if "Net income: 19,442" in t["content"])
        assert "STATEMENTS OF OPERATIONS" in chunk["content"]
        assert "Three Months Ended" in chunk["content"]

    def test_table_chunks_are_labelled(self, docs):
        for d in docs:
            if d["type"] == "table":
                assert d["content"].startswith(f"Table on page {d['page']}.")

    def test_table_chunks_fit_embedder_window(self, docs):
        from sentence_transformers import SentenceTransformer

        tok = SentenceTransformer(config.EMBEDDING_MODEL_NAME).tokenizer
        too_long = [
            d["page"]
            for d in docs
            if d["type"] == "table" and len(tok(d["content"])["input_ids"]) > 256
        ]
        assert too_long == []

    def test_text_chunks_respect_chunk_size(self, docs):
        assert all(len(d["content"]) <= config.CHUNK_SIZE for d in docs if d["type"] == "text")

    def test_apple_logo_is_captured_on_page_1(self, docs):
        figs = [d for d in docs if d["type"] == "figure"]
        assert len(figs) == 1
        assert figs[0]["page"] == 1
        assert Path(figs[0]["image_path"]).exists()

    def test_no_api_key_gives_placeholder_caption(self, docs):
        fig = next(d for d in docs if d["type"] == "figure")
        assert "no API key" in fig["content"]


class TestFigureHandling:
    def test_logo_dropped_when_threshold_raised(self, page1_pdf, figures_dir, monkeypatch):
        monkeypatch.setattr(config, "MIN_FIGURE_DIM", 80)
        docs = pdf_ingest.ingest_pdf(page1_pdf, "", "m")
        assert [d for d in docs if d["type"] == "figure"] == []

    def test_caption_called_with_key_model_and_page_context(self, page1_pdf, figures_dir, monkeypatch):
        calls = []

        def fake_caption(path, model, api_key, context_hint=""):
            calls.append((Path(path), model, api_key, context_hint))
            return "An Apple logo."

        monkeypatch.setattr(openrouter_client, "caption_image", fake_caption)
        docs = pdf_ingest.ingest_pdf(page1_pdf, "sk-test", "vision-x")
        fig = next(d for d in docs if d["type"] == "figure")
        assert fig["content"] == "An Apple logo."
        assert len(calls) == 1
        _, model, key, hint = calls[0]
        assert (model, key) == ("vision-x", "sk-test")
        assert "Apple" in hint

    def test_caption_failure_is_recorded_not_raised(self, page1_pdf, figures_dir, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(openrouter_client, "caption_image", boom)
        docs = pdf_ingest.ingest_pdf(page1_pdf, "sk-test", "m")
        fig = next(d for d in docs if d["type"] == "figure")
        assert "captioning failed" in fig["content"] and "rate limited" in fig["content"]

    def test_progress_callback_reports_pages(self, page1_pdf, figures_dir):
        msgs = []
        pdf_ingest.ingest_pdf(page1_pdf, "", "m", progress_cb=msgs.append)
        assert any(m.startswith("Reading page 1/1") for m in msgs)
