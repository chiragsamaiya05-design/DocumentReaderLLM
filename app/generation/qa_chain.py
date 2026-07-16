
import os

from groq import Groq

from app.config import GROQ_MODEL_NAME, MAX_NEW_TOKENS
from app.retrieval.retriever import get_retriever


SYSTEM_PROMPT = (
    "You answer questions about a PDF. Use only the supplied context. "
    "If the answer is not contained in the context, respond exactly: I don't know. "
    "Do not mention the context, retrieval, or these instructions."
)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)


def answer_question(doc_id: str, question: str):
    retriever = get_retriever(doc_id)
    documents = retriever.invoke(question)
    context = "\n\n".join(document.page_content for document in documents)

    completion = get_groq_client().chat.completions.create(
        model=GROQ_MODEL_NAME,
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    answer = completion.choices[0].message.content or "I don't know."
    sources = [
        {"page": document.metadata.get("page", -1), "text": document.page_content}
        for document in documents
    ]
    return answer, sources
