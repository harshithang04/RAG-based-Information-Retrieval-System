import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = DATA_DIR / "figures"
INDEX_PATH = DATA_DIR / "index.npz"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free OpenRouter model slugs churn often - verify at
# https://openrouter.ai/models?max_price=0 if a request starts failing.
TEXT_MODEL = os.getenv("TEXT_MODEL", "inclusionai/ling-3.0-flash-fin:free")
VISION_MODEL = os.getenv("VISION_MODEL", "inclusionai/ling-3.0-flash-vl:free")

# Ling-3.0-flash-Fin is a "thinking" model: it reasons in a separate
# `reasoning` field before writing `content`, and its model card recommends
# temperature=1.0/top_p=0.95/top_k=20. max_tokens must cover the reasoning
# budget too, or content comes back null when it gets cut off mid-thought.
TEXT_TEMPERATURE = 1.0
TEXT_TOP_P = 0.95
TEXT_TOP_K = 20
TEXT_MAX_TOKENS = 2000

TEXT_MODEL_CHOICES = [
    "inclusionai/ling-3.0-flash-fin:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/free",
]
VISION_MODEL_CHOICES = [
    "inclusionai/ling-3.0-flash-vl:free",
    "thinkingmachines/inkling:free",
    "thinkingmachines/inkling-small:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_FIGURE_DIM = 40  # px, filters out tiny icons/bullets (keeps small logos, e.g. 46x56 Apple logo)
TABLE_CHUNK_CHARS = 200  # row text per table chunk; with context it must fit the embedder's 256-token window
TABLE_CONTEXT_CHARS = 300  # text above a table (title, column headers) repeated in each chunk
TOP_K = 5
HYBRID_CANDIDATES = 50  # per-retriever candidates fused (RRF) before taking top_k
RRF_K = 60
