import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
# Storage configuration
DATA_DIR = Path(
    os.getenv("DATA_DIR", str(BASE_DIR / "data"))
)

UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indexes"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)


TOP_K = 5
RETRIEVAL_CANDIDATES = 20
LEXICAL_CANDIDATES = 8


MAX_UPLOAD_BYTES = 20 * 1024 * 1024


GROQ_MODEL_NAME = os.getenv(
    "GROQ_MODEL_NAME",
    "openai/gpt-oss-120b"
)
MAX_NEW_TOKENS = 2048

RELEVANCE_THRESHOLD = float(
    os.getenv("RELEVANCE_THRESHOLD", "0.0")
)