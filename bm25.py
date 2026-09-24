import math
import re

import numpy as np

# Words, plus numbers keeping their thousands separators/decimals ("82,959", "4.82"),
# so exact figures in financial tables are matchable.
_TOKEN = re.compile(r"[a-z]+|\d[\d,\.]*\d|\d")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Small Okapi BM25 index - fine at the scale of a handful of PDFs."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in texts]
        self.lengths = [len(d) for d in docs]
        self.avg_len = (sum(self.lengths) / len(docs)) if docs else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for d in docs:
            counts: dict[str, int] = {}
            for w in d:
                counts[w] = counts.get(w, 0) + 1
            self.tf.append(counts)
            for w in counts:
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        terms = tokenize(query)
        for i, counts in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1.0))
            for w in terms:
                f = counts.get(w)
                if f:
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return out
