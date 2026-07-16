
from app.retrieval.vector_store import load_index
from app.config import TOP_K


def get_retriever(doc_id: str):
    vectorstore = load_index(doc_id)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
