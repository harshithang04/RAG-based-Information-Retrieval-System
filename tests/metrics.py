"""Retrieval quality metrics. `relevance` is a list of bools, one per retrieved chunk, in rank order."""


def _norm(s: str) -> str:
    return " ".join(s.split())


def is_relevant(chunk: dict, evidence: list[str]) -> bool:
    content = _norm(chunk["content"])
    return all(_norm(e) in content for e in evidence)


def hit_at_k(relevance: list[bool], k: int) -> float:
    return float(any(relevance[:k]))


def precision_at_k(relevance: list[bool], k: int) -> float:
    top = relevance[:k]
    return sum(top) / len(top) if top else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
