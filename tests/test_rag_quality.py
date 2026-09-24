"""End-to-end RAG quality metrics on the golden set.

Run `pytest tests/test_rag_quality.py -s` to print the report. The report is also
written to tests/metrics_report.json. Thresholds are regression floors: if a change
to chunking, embedding model or retrieval drops a metric below them, the test fails.
"""

import json
import os
import time
from pathlib import Path

import pytest
import requests

import config
import openrouter_client
from tests.golden_set import GOLDEN_SET
from tests.metrics import hit_at_k, is_relevant, mean, precision_at_k, reciprocal_rank

REPORT_PATH = Path(__file__).parent / "metrics_report.json"
K_VALUES = (1, 3, 5)

# Regression floors on the held-out split (the honest measure - retrieval was tuned on "dev").
# History (hit@5 / hit@3 / MRR):
#   dense-only baseline, dev:                0.71 / 0.57 / 0.43
#   hybrid + row-level table chunks, dev:    1.00 / 0.71 / 0.66   (tuned on this split)
#   hybrid + row-level table chunks, heldout 0.83 / 0.67 / 0.61   (not tuned on)
# Floors are set just under the held-out values; raise them as retrieval improves.
MIN_HIT_AT_5 = 0.80
MIN_HIT_AT_3 = 0.60
MIN_MRR = 0.55


def _summarise(rows: list[dict]) -> dict:
    by_type = {}
    for row in rows:
        first = row["retrieved"][row["rank_of_first_hit"] - 1][0] if row["rank_of_first_hit"] else "miss"
        by_type[first] = by_type.get(first, 0) + 1
    return {
        "n_questions": len(rows),
        "hit@1": mean([r["hit@1"] for r in rows]),
        "hit@3": mean([r["hit@3"] for r in rows]),
        "hit@5": mean([r["hit@5"] for r in rows]),
        "precision@5": mean([r["precision@5"] for r in rows]),
        "mrr": mean([r["rr"] for r in rows]),
        "first_hit_chunk_type": by_type,
    }


@pytest.fixture(scope="module")
def retrieval_report(indexed_pipeline):
    per_case = []
    for case in GOLDEN_SET:
        results = indexed_pipeline.retrieve(case["question"], top_k=max(K_VALUES))
        rel = [is_relevant(r, case["evidence"]) for r in results]
        per_case.append(
            {
                "id": case["id"],
                "split": case["split"],
                "question": case["question"],
                "rank_of_first_hit": next((i for i, r in enumerate(rel, 1) if r), None),
                "hit@1": hit_at_k(rel, 1),
                "hit@3": hit_at_k(rel, 3),
                "hit@5": hit_at_k(rel, 5),
                "precision@5": precision_at_k(rel, 5),
                "rr": reciprocal_rank(rel),
                "retrieved": [(r["type"], r["page"], round(r["score"], 3)) for r in results],
            }
        )
    report = {
        "config": {
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "table_chunk_chars": config.TABLE_CHUNK_CHARS,
            "top_k": config.TOP_K,
            "hybrid": "dense + BM25, RRF",
        },
        "summary": {
            "all": _summarise(per_case),
            "dev": _summarise([c for c in per_case if c["split"] == "dev"]),
            "heldout": _summarise([c for c in per_case if c["split"] == "heldout"]),
            "extra": _summarise([c for c in per_case if c["split"] == "extra"]),
        },
        "cases": per_case,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def test_print_report(retrieval_report):
    print("\n=== RAG retrieval metrics ===")
    for split, s in retrieval_report["summary"].items():
        print(f"[{split}] n={s['n_questions']} hit@1={s['hit@1']:.2f} hit@3={s['hit@3']:.2f} "
              f"hit@5={s['hit@5']:.2f} P@5={s['precision@5']:.2f} MRR={s['mrr']:.2f} "
              f"first-hit types={s['first_hit_chunk_type']}")
    print("--- per question (rank of first relevant chunk) ---")
    for c in retrieval_report["cases"]:
        print(f"{c['split']:<8}{c['id']:<28} rank={c['rank_of_first_hit']}")
    misses = [c["id"] for c in retrieval_report["cases"] if c["rank_of_first_hit"] is None]
    if misses:
        print("MISSES:", ", ".join(misses))


def test_golden_evidence_exists_in_corpus(docs):
    """Guards the eval itself: every golden case must be answerable from some chunk."""
    for case in GOLDEN_SET:
        assert any(is_relevant(d, case["evidence"]) for d in docs), f"no chunk holds {case['id']}"


def test_heldout_hit_at_5_floor(retrieval_report):
    assert retrieval_report["summary"]["heldout"]["hit@5"] >= MIN_HIT_AT_5


def test_heldout_hit_at_3_floor(retrieval_report):
    assert retrieval_report["summary"]["heldout"]["hit@3"] >= MIN_HIT_AT_3


def test_heldout_mrr_floor(retrieval_report):
    assert retrieval_report["summary"]["heldout"]["mrr"] >= MIN_MRR


def test_dev_hit_at_5_floor(retrieval_report):
    assert retrieval_report["summary"]["dev"]["hit@5"] >= MIN_HIT_AT_5


def test_hits_are_monotonic_in_k(retrieval_report):
    s = retrieval_report["summary"]["all"]
    assert s["hit@1"] <= s["hit@3"] <= s["hit@5"]


# ---- Generation quality: needs a real LLM, so opt-in ----------------------------------

MIN_ANSWER_ACCURACY = 0.6


@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("RUN_LIVE_TESTS") and os.getenv("OPENROUTER_API_KEY")),
    reason="opt in with RUN_LIVE_TESTS=1 and OPENROUTER_API_KEY (slow, uses the network)",
)
def test_live_answer_accuracy_and_citations(indexed_pipeline):
    key = os.environ["OPENROUTER_API_KEY"]
    correct, cited, errors, infra_errors, total = 0, 0, 0, 0, 0
    print()
    live_cases = [c for c in GOLDEN_SET if c["split"] != "extra"]  # 26 calls: fits the free 50/day
    for n, case in enumerate(live_cases):
        if n:
            time.sleep(4)  # stay under the free tier's per-minute rate limit
        try:
            text, _ = indexed_pipeline.answer(case["question"], key, config.TEXT_MODEL)
        except openrouter_client.OpenRouterError as e:
            if "error 429" in str(e):  # rate limited: says nothing about the pipeline
                infra_errors += 1
                print(f"{case['id']:<26} SKIPPED (rate limited)")
                continue
            total += 1  # e.g. thinking model ran out of tokens: a failed answer
            errors += 1
            print(f"{case['id']:<26} ERROR  {str(e)[:80]}")
            continue
        except requests.RequestException as e:
            infra_errors += 1
            print(f"{case['id']:<26} SKIPPED ({type(e).__name__})")
            continue
        total += 1
        ok = case["answer"].replace(",", "") in text.replace(",", "")
        correct += ok
        cited += "page" in text.lower()
        print(f"{case['id']:<26} {'OK   ' if ok else 'WRONG'}  {text[:90]!r}")
    if total == 0:
        pytest.skip("OpenRouter unavailable for every question")
    accuracy, citation_rate = correct / total, cited / total
    print(f"answer accuracy={accuracy:.2f} ({correct}/{total}) citation_rate={citation_rate:.2f} "
          f"model_errors={errors} excluded_infra_errors={infra_errors}")
    assert accuracy >= MIN_ANSWER_ACCURACY
