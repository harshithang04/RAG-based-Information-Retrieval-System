import os
from pathlib import Path

# `import app` builds a Gradio Blocks, which otherwise sends analytics on import.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import numpy as np
import pymupdf
import pytest

import config
import pdf_ingest
from rag_pipeline import RAGPipeline

PDF_PATH = config.INPUT_DIR / "2022 Q3 AAPL.pdf"


@pytest.fixture(scope="session")
def pdf_path() -> Path:
    if not PDF_PATH.exists():
        pytest.skip(f"sample PDF missing: {PDF_PATH}")
    return PDF_PATH


@pytest.fixture(scope="session")
def figures_dir(tmp_path_factory):
    """Redirect extracted figures away from the real data/ dir for the whole session."""
    d = tmp_path_factory.mktemp("figures")
    original = config.FIGURES_DIR
    config.FIGURES_DIR = d
    yield d
    config.FIGURES_DIR = original


@pytest.fixture(scope="session")
def docs(pdf_path, figures_dir) -> list[dict]:
    """Full ingestion of the sample PDF without an API key (no network)."""
    return pdf_ingest.ingest_pdf(pdf_path, api_key="", vision_model="unused")


@pytest.fixture(scope="session")
def page1_pdf(pdf_path, tmp_path_factory) -> Path:
    """One-page slice (contains the logo) so ingestion tests stay fast."""
    out = tmp_path_factory.mktemp("slice") / "page1.pdf"
    src = pymupdf.open(pdf_path)
    dst = pymupdf.open()
    dst.insert_pdf(src, from_page=0, to_page=0)
    dst.save(out)
    return out


@pytest.fixture(scope="session")
def indexed_pipeline(docs) -> RAGPipeline:
    """Pipeline with the sample PDF embedded in memory. Never touches data/index.npz."""
    pipe = RAGPipeline()
    embeddings = pipe.embedder.encode(
        [d["content"] for d in docs], batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )
    pipe.store.build(np.asarray(embeddings, dtype=np.float32), docs)
    return pipe
