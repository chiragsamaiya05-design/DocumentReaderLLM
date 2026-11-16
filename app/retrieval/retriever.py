import re

from collections import Counter
from typing import Iterable, List, Tuple

from langchain.schema import Document


from app.retrieval.vector_store import (
    load_index,
    index_exists
)

from app.config import (
    LEXICAL_CANDIDATES,
    RETRIEVAL_CANDIDATES,
    TOP_K,
    RELEVANCE_THRESHOLD
)


# ============================================================
# TOKENIZATION CONFIGURATION
# ============================================================

_TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}"
)

_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are",
    "been", "before", "being", "between", "could",
    "does", "from", "have", "into", "more", "most",
    "only", "other", "should", "than", "that",
    "the", "their", "there", "these", "this",
    "those", "through", "under", "what", "when",
    "where", "which", "with", "would",
}


# ============================================================
# TOKENIZATION
# ============================================================

def _tokens(text: str) -> List[str]:

    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if token.lower() not in _STOP_WORDS
    ]


# ============================================================
# LEXICAL SCORING
# ============================================================

def _lexical_score(
    question_tokens: Iterable[str],
    document: Document
) -> float:

    query = list(question_tokens)

    if not query:
        return 0.0

    counts = Counter(
        _tokens(document.page_content)
    )

    return (
        sum(
            min(counts[token], 2)
            for token in set(query)
        )
        / len(set(query))
    )


# ============================================================
# FETCH ALL DOCUMENTS
# ============================================================

def _all_documents(vectorstore) -> List[Document]:

    return list(
        vectorstore.docstore._dict.values()
    )


# ============================================================
# DOCUMENT IDENTIFICATION
# ============================================================

def _document_key(document: Document) -> tuple:

    return (
        document.metadata.get("source"),
        document.metadata.get("chunk_id"),
        document.page_content
    )


# ============================================================
# RETRIEVAL WITH DEBUG INFORMATION
# ============================================================

def retrieve_with_debug(
    doc_ids,
    question: str
) -> Tuple[List[Document], dict]:

    # --------------------------------------------------------
    # STEP 1: INITIALIZE STORAGE
    # --------------------------------------------------------

    candidates = {}
    semantic_scores = {}
    dense_ranked = []
    dense_ranks = {}
    lexical_ranks = {}
    documents = {}
    scores = {}
    document_groups = {}

    debug_info = {
        "query": question,
        "retrieval_strategy": "Document-Aware Hybrid Retrieval",
        "dense_candidates": 0,
        "lexical_candidates": 0,
        "final_chunks": 0,
        "selected_documents": len(doc_ids),
        "document_coverage": {},
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "relevant_chunks": 0,
        "rejected_chunks": 0,
        "retrieved_chunks": []
    }

    # --------------------------------------------------------
    # STEP 2: SEMANTIC SCORING AND DENSE RETRIEVAL
    # --------------------------------------------------------

    for doc_id in doc_ids:

        if not index_exists(doc_id):
            continue

        vectorstore = load_index(doc_id)

        # Score every chunk in this PDF.
        results = vectorstore.similarity_search_with_relevance_scores(
            question,
            k=vectorstore.index.ntotal
        )

        for document, relevance_score in results:

            key = _document_key(document)

            candidates[key] = document
            documents[key] = document
            semantic_scores[key] = float(relevance_score)

            document_groups[key] = (
                document.metadata.get("source")
                or document.metadata.get("filename")
                or doc_id
            )

    # Select globally highest-scoring dense candidates.
    dense_sorted = sorted(
        semantic_scores,
        key=semantic_scores.get,
        reverse=True
    )

    dense_keys = dense_sorted[:RETRIEVAL_CANDIDATES]

    for rank, key in enumerate(dense_keys, start=1):

        dense_ranked.append((rank, key))
        dense_ranks[key] = rank

    debug_info["dense_candidates"] = len(dense_ranked)

    # --------------------------------------------------------
    # STEP 3: LEXICAL RETRIEVAL
    # --------------------------------------------------------

    question_tokens = _tokens(question)

    lexical_sorted = sorted(
        candidates,
        key=lambda key: _lexical_score(
            question_tokens,
            candidates[key]
        ),
        reverse=True
    )

    lexical_keys = lexical_sorted[:LEXICAL_CANDIDATES]

    for rank, key in enumerate(lexical_keys, start=1):
        lexical_ranks[key] = rank

    debug_info["lexical_candidates"] = len(lexical_keys)

    # --------------------------------------------------------
    # STEP 4: RECIPROCAL RANK FUSION
    # --------------------------------------------------------

    RRF_CONSTANT = 60

    for rank, key in dense_ranked:

        scores[key] = (
            scores.get(key, 0.0)
            + 1 / (RRF_CONSTANT + rank)
        )

    for rank, key in enumerate(lexical_keys, start=1):

        scores[key] = (
            scores.get(key, 0.0)
            + 1 / (RRF_CONSTANT + rank)
        )

    # --------------------------------------------------------
    # STEP 5: GLOBAL RRF RANKING
    # --------------------------------------------------------

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    # --------------------------------------------------------
    # STEP 6: RELEVANCE FILTERING
    # --------------------------------------------------------

    relevant_keys = [
        key
        for key in ranked_keys
        if semantic_scores.get(key, 0.0) >= RELEVANCE_THRESHOLD
    ]

    rejected_keys = [
        key
        for key in ranked_keys
        if semantic_scores.get(key, 0.0) < RELEVANCE_THRESHOLD
    ]

    debug_info["relevant_chunks"] = len(relevant_keys)
    debug_info["rejected_chunks"] = len(rejected_keys)

    # --------------------------------------------------------
    # STEP 7: DOCUMENT-AWARE SELECTION
    # --------------------------------------------------------

    selected_keys = []
    selected_groups = set()

    # Reserve the best relevant chunk from each PDF.
    if len(doc_ids) <= TOP_K:

        for key in relevant_keys:

            group = document_groups.get(key, "unknown")

            if group not in selected_groups:

                selected_keys.append(key)
                selected_groups.add(group)

    # Fill remaining slots using RRF ranking.
    for key in relevant_keys:

        if len(selected_keys) >= TOP_K:
            break

        if key not in selected_keys:
            selected_keys.append(key)

    # --------------------------------------------------------
    # STEP 8: BUILD FINAL DOCUMENTS AND DEBUG INFORMATION
    # --------------------------------------------------------

    final_documents = []
    coverage = {}

    for key in selected_keys:

        document = documents[key]

        final_documents.append(document)

        filename = document.metadata.get(
            "filename",
            "unknown"
        )

        coverage[filename] = (
            coverage.get(filename, 0) + 1
        )

        methods = []

        if key in dense_ranks:
            methods.append("dense")

        if key in lexical_ranks:
            methods.append("lexical")

        debug_info["retrieved_chunks"].append({

            "filename": filename,

            "page": document.metadata.get("page", -1),

            "chunk_id": document.metadata.get("chunk_id", -1),

            "dense_rank": dense_ranks.get(key),

            "lexical_rank": lexical_ranks.get(key),

            "rrf_score": round(scores[key], 6),

            "semantic_relevance": round(
                semantic_scores[key], 4
            ),

            "retrieval_methods": methods,

            "text": document.page_content

        })

    debug_info["document_coverage"] = coverage
    debug_info["final_chunks"] = len(final_documents)

    return final_documents, debug_info
    
# ============================================================
# BACKWARD-COMPATIBLE RETRIEVAL FUNCTION
# ============================================================

def retrieve(
    doc_ids,
    question: str
) -> List[Document]:

    documents, _ = retrieve_with_debug(
        doc_ids,
        question
    )

    return documents