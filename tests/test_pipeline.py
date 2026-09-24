import pytest

import config
import openrouter_client
import pdf_ingest
import rag_pipeline
from rag_pipeline import RAGPipeline


@pytest.fixture
def isolated_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "index.npz")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    return RAGPipeline()


def test_answer_before_ingest_prompts_upload(isolated_pipeline):
    text, results = isolated_pipeline.answer("anything", "key", "model")
    assert "Upload a PDF" in text and results == []


def test_retrieve_before_ingest_is_empty(isolated_pipeline):
    assert isolated_pipeline.retrieve("anything") == []


def test_ingest_builds_persists_and_reloads(isolated_pipeline, page1_pdf, tmp_path):
    counts = isolated_pipeline.ingest(page1_pdf, "", "m")
    assert counts["text"] > 0 and counts["figure"] == 1
    assert config.INDEX_PATH.exists() and config.INDEX_PATH.with_suffix(".meta.json").exists()

    reloaded = RAGPipeline()
    assert reloaded.store.is_ready()
    assert len(reloaded.store.metadatas) == sum(counts.values())


def test_ingest_empty_pdf_raises(isolated_pipeline, page1_pdf, monkeypatch):
    monkeypatch.setattr(pdf_ingest, "ingest_pdf", lambda *a, **k: [])
    with pytest.raises(ValueError):
        isolated_pipeline.ingest(page1_pdf, "", "m")


def test_retrieve_respects_top_k_and_orders_by_score(indexed_pipeline):
    res = indexed_pipeline.retrieve("total net sales", top_k=3)
    assert len(res) == 3
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_answer_prompt_contains_tagged_context_and_model_settings(indexed_pipeline, monkeypatch):
    captured = {}

    def fake_chat(messages, model, api_key, **kwargs):
        captured.update(messages=messages, model=model, api_key=api_key, **kwargs)
        return "stub answer"

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat)
    q = "What was iPhone net sales in Q3 2022?"
    text, results = indexed_pipeline.answer(q, "sk", "text-model", top_k=4)

    assert text == "stub answer" and len(results) == 4
    assert captured["model"] == "text-model" and captured["api_key"] == "sk"
    assert captured["max_tokens"] == config.TEXT_MAX_TOKENS
    system, user = captured["messages"]
    assert system["content"] == rag_pipeline.SYSTEM_PROMPT
    assert q in user["content"]
    assert user["content"].count("[") >= 4
    for r in results:
        assert f"[{r['type'].upper()} | page {r['page']}]" in user["content"]
        assert r["content"] in user["content"]
