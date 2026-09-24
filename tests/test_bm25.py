import numpy as np

from bm25 import BM25, tokenize
from vector_store import VectorStore


def test_tokenize_keeps_formatted_numbers():
    assert tokenize("Net sales were $82,959 million, EPS 4.82!") == [
        "net", "sales", "were", "82,959", "million", "eps", "4.82",
    ]


def test_exact_number_ranks_matching_doc_first():
    bm = BM25(["revenue was 82,959 million", "revenue was 81,434 million", "no figures here"])
    assert int(np.argmax(bm.scores("what is 82,959"))) == 0


def test_rare_term_outweighs_common_term():
    bm = BM25(["the the the company", "the dividend", "the the company company"])
    assert int(np.argmax(bm.scores("the dividend"))) == 1


def test_no_overlap_scores_zero():
    assert not BM25(["alpha beta", "gamma"]).scores("zzz").any()


def test_empty_corpus_does_not_crash():
    assert len(BM25([]).scores("x")) == 0


def _store():
    """Docs 0-2 are dense-identical decoys; only doc 3 has the exact figure."""
    texts = ["general remarks", "more general remarks", "other remarks", "total net sales 82,959"]
    emb = np.array([[1, 0], [0.99, 0.1], [0.98, 0.2], [0.5, 0.5]], dtype=np.float32)
    s = VectorStore()
    s.build(emb, [{"content": t, "type": "text", "page": i} for i, t in enumerate(texts)])
    return s


def test_hybrid_surfaces_exact_match_that_dense_ranks_last():
    s = _store()
    q = np.array([1, 0], dtype=np.float32)
    assert s.search(q, top_k=4)[-1]["content"] == "total net sales 82,959"  # dense buries it
    assert s.hybrid_search(q, "82,959", top_k=2)[0]["content"] == "total net sales 82,959"


def test_hybrid_results_sorted_and_carry_both_scores():
    res = _store().hybrid_search(np.array([1, 0], dtype=np.float32), "remarks", top_k=4)
    assert [r["score"] for r in res] == sorted((r["score"] for r in res), reverse=True)
    assert all("dense_score" in r for r in res)


def test_hybrid_without_lexical_match_falls_back_to_dense():
    res = _store().hybrid_search(np.array([1, 0], dtype=np.float32), "zzz", top_k=2)
    assert res[0]["content"] == "general remarks"


def test_hybrid_top_k_clamped_and_empty_store():
    assert len(_store().hybrid_search(np.array([1, 0], dtype=np.float32), "remarks", top_k=50)) <= 4
    assert VectorStore().hybrid_search(np.array([1, 0], dtype=np.float32), "x") == []


def test_bm25_rebuilt_after_load(tmp_path):
    s = _store()
    s.save(tmp_path / "i.npz")
    loaded = VectorStore()
    loaded.load(tmp_path / "i.npz")
    q = np.array([1, 0], dtype=np.float32)
    assert loaded.hybrid_search(q, "82,959", 2) == s.hybrid_search(q, "82,959", 2)
