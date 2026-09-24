import numpy as np

from vector_store import VectorStore


def make_store():
    emb = np.array([[1, 0, 0], [0, 1, 0], [0.7, 0.7, 0], [0, 0, 1]], dtype=np.float32)
    metas = [{"content": c} for c in "abcd"]
    s = VectorStore()
    s.build(emb, metas)
    return s


def test_empty_store_not_ready_and_returns_nothing():
    s = VectorStore()
    assert not s.is_ready()
    assert s.search(np.array([1.0, 0.0, 0.0])) == []


def test_exact_match_ranks_first_with_score_one():
    res = make_store().search(np.array([0, 1, 0], dtype=np.float32), top_k=3)
    assert res[0]["content"] == "b"
    assert abs(res[0]["score"] - 1.0) < 1e-6


def test_results_sorted_descending():
    res = make_store().search(np.array([1, 0.2, 0], dtype=np.float32), top_k=4)
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_top_k_clamped_to_store_size():
    assert len(make_store().search(np.array([1, 0, 0], dtype=np.float32), top_k=50)) == 4


def test_query_scale_invariant():
    s = make_store()
    a = s.search(np.array([1, 1, 0], dtype=np.float32), top_k=2)
    b = s.search(np.array([10, 10, 0], dtype=np.float32), top_k=2)
    assert [r["content"] for r in a] == [r["content"] for r in b]


def test_embeddings_normalised_and_zero_vector_safe():
    s = VectorStore()
    s.build(np.array([[3, 4], [0, 0]], dtype=np.float32), [{"content": "x"}, {"content": "z"}])
    assert np.allclose(np.linalg.norm(s.embeddings[0]), 1.0)
    assert not np.isnan(s.embeddings).any()


def test_search_does_not_mutate_metadata():
    s = make_store()
    s.search(np.array([1, 0, 0], dtype=np.float32))
    assert "score" not in s.metadatas[0]


def test_save_load_roundtrip(tmp_path):
    s = make_store()
    path = tmp_path / "idx.npz"
    s.save(path)
    loaded = VectorStore()
    assert loaded.load(path)
    assert loaded.metadatas == s.metadatas
    q = np.array([0.7, 0.7, 0], dtype=np.float32)
    assert loaded.search(q, 2) == s.search(q, 2)


def test_load_missing_returns_false(tmp_path):
    assert VectorStore().load(tmp_path / "nope.npz") is False
