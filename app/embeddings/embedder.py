"""
Wraps LangChain's HuggingFaceEmbeddings so the same embedding object
can be passed directly into a LangChain vector store (FAISS), instead
of manually calling .encode() and managing numpy arrays ourselves.
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL_NAME

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings
