"""
Builds and persists a LangChain FAISS vector store per document.
LangChain's FAISS wrapper handles the index + docstore + id-mapping
internally (previously we managed a sidecar JSON metadata file by hand).
"""
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from app.config import INDEX_DIR
from app.embeddings.embedder import get_embeddings


def _index_dir(doc_id: str) -> Path:
    return INDEX_DIR / doc_id


def build_index(doc_id: str, chunks: List[Document]) -> None:
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(_index_dir(doc_id)))


def load_index(doc_id: str) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(_index_dir(doc_id)),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we created this index ourselves
    )


def index_exists(doc_id: str) -> bool:
    return _index_dir(doc_id).exists()
