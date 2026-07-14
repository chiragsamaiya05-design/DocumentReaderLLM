import os
from pathlib import Path
from dotenv import load_dotenv

# --- Base paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indexes"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- Chunking (LangChain's RecursiveCharacterTextSplitter) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 4

# --- Generation ---
# Groq hosts the LLM, so the application does not download a local generation
# model. Set GROQ_API_KEY in the environment before starting the API.
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
MAX_NEW_TOKENS = 256
