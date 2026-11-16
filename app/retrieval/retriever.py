
import re
from collections import Counter
from typing import Iterable, List

from langchain.schema import Document

from app.config import LEXICAL_CANDIDATES, RETRIEVAL_CANDIDATES, TOP_K
from app.retrieval.vector_store import load_index, index_exists

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}")
_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "been", "before", "being",
    "between", "could", "does", "from", "have", "into", "more", "most", "only",
    "other", "should", "than", "that", "the", "their", "there", "these", "this",
    "those", "through", "under", "what", "when", "where", "which", "with", "would",
}


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text) if token.lower() not in _STOP_WORDS]


def _lexical_score(question_tokens: Iterable[str], document: Document) -> float:
    query = list(question_tokens)
    if not query:
        return 0.0
    counts = Counter(_tokens(document.page_content))
    # Repeated terms have diminishing returns; this favours chunks that cover
    # more of the user's specific terms instead of a single repeated word.
    return sum(min(counts[token], 2) for token in set(query)) / len(set(query))


def _all_documents(vectorstore) -> List[Document]:
    # FAISS's in-memory docstore is persisted with the index and avoids an
    # extra dependency for exact-term retrieval.
    return list(vectorstore.docstore._dict.values())


def retrieve(doc_ids, question: str) -> List[Document]:
    candidates = []
    dense_ranked = []
    for doc_id in doc_ids:
        if not index_exists(doc_id):
            continue
        vectorstore = load_index(doc_id)
        dense = vectorstore.max_marginal_relevance_search(
            question,
            k=RETRIEVAL_CANDIDATES,
            fetch_k=max(RETRIEVAL_CANDIDATES * 2, 30),
            lambda_mult=0.7,
        )
        candidates.extend(_all_documents(vectorstore))
        for rank, document in enumerate(dense, start=1):
            dense_ranked.append((rank, document))
    question_tokens = _tokens(question)
    lexical = sorted(
        candidates,
        key=lambda doc: _lexical_score(question_tokens, doc),
        reverse=True,
    )[:LEXICAL_CANDIDATES]

    # Reciprocal-rank fusion is robust when either semantic or exact-match
    # retrieval is the better signal for a question.
    scores = {}
    documents = {}
    for rank, document in dense_ranked:
        key = (document.metadata.get("source"), document.metadata.get("chunk_id"), document.page_content)
        documents[key] = document
        scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
    for rank, document in enumerate(lexical, start=1):
        key = (document.metadata.get("source"), document.metadata.get("chunk_id"), document.page_content)
        documents[key] = document
        scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)

    return [documents[key] for key in sorted(scores, key=scores.get, reverse=True)[:TOP_K]]
