import json
from pathlib import Path

import numpy as np

import config
from bm25 import BM25


class VectorStore:
    """Minimal in-memory cosine-similarity store, persisted to a single .npz file.

    Avoids a faiss/chroma dependency - fine at the scale of a handful of PDFs.
    """

    def __init__(self):
        self.embeddings: np.ndarray | None = None  # (n, dim), L2-normalized
        self.metadatas: list[dict] = []
        self._bm25: BM25 | None = None

    def build(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = embeddings / norms
        self.metadatas = metadatas
        self._bm25 = BM25([m["content"] for m in metadatas])

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        candidates: int = config.HYBRID_CANDIDATES,
        rrf_k: int = config.RRF_K,
    ) -> list[dict]:
        """Fuse dense (cosine) and BM25 rankings with Reciprocal Rank Fusion.

        Dense search handles paraphrase; BM25 handles exact figures and rare phrases
        ("82,959", "408 million shares") that small embedding models blur.
        `score` is the fused RRF score (results are sorted by it); `dense_score` is the cosine.
        """
        if self.embeddings is None or len(self.metadatas) == 0:
            return []
        q = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
        dense = self.embeddings @ q
        sparse = self._bm25.scores(query_text)
        fused: dict[int, float] = {}
        for ranking, scores in (("dense", dense), ("bm25", sparse)):
            order = np.argsort(-scores)[:candidates]
            for rank, i in enumerate(order):
                if ranking == "bm25" and scores[i] <= 0:
                    break  # no query term matched: no lexical evidence
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[: min(top_k, len(self.metadatas))]
        results = []
        for i in best:
            item = dict(self.metadatas[i])
            item["score"] = fused[i]
            item["dense_score"] = float(dense[i])
            results.append(item)
        return results

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.embeddings is None or len(self.metadatas) == 0:
            return []
        q = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
        scores = self.embeddings @ q
        top_k = min(top_k, len(self.metadatas))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        results = []
        for i in idx:
            item = dict(self.metadatas[i])
            item["score"] = float(scores[i])
            results.append(item)
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        np.savez_compressed(path, embeddings=self.embeddings)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(self.metadatas, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> bool:
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        if not path.exists() or not meta_path.exists():
            return False
        with np.load(path) as data:
            self.embeddings = data["embeddings"]
        self.metadatas = json.loads(meta_path.read_text(encoding="utf-8"))
        self._bm25 = BM25([m["content"] for m in self.metadatas])
        return True

    def is_ready(self) -> bool:
        return self.embeddings is not None and len(self.metadatas) > 0
