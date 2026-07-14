"""
Turns a loaded FAISS vector store into a LangChain retriever object,
which the RetrievalQA chain can call directly — no manual similarity
search or numpy handling needed on our side anymore.
"""
from app.retrieval.vector_store import load_index
from app.config import TOP_K


def get_retriever(doc_id: str):
    vectorstore = load_index(doc_id)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
