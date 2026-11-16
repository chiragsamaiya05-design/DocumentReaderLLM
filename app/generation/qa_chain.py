import os
import logging
import time

from groq import Groq

from app.config import GROQ_MODEL_NAME, MAX_NEW_TOKENS
from app.retrieval.retriever import retrieve


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are DocuMind, an intelligent document question-answering
assistant specializing in accurate document-grounded responses.

Your task is to answer questions using the retrieved document
passages provided to you.

CORE INSTRUCTIONS:

1. DOCUMENT GROUNDING

- Use ONLY information contained in the provided document context.
- Never introduce external knowledge or unsupported facts.
- Preserve the terminology and meaning of the original document.

2. MULTI-PASSAGE REASONING

- Carefully examine ALL retrieved passages.
- Information may be distributed across multiple passages.
- Combine relevant information from different passages.
- Do not evaluate each passage independently.
- For comparison questions, identify similarities and differences.
- For multi-part questions, address each part separately.
- Use examples when they are available in the context.

3. PARTIAL INFORMATION

- If some information is available, answer using that information.
- Clearly identify aspects that cannot be established.
- Do not reject an entire question merely because one detail
  is missing from the retrieved context.

4. ABSTENTION

- If the retrieved context contains no relevant information,
  respond:

  "I couldn't find sufficient information about this
  in the uploaded documents."

- Do not use "I don't know" when relevant evidence is available.
- Never invent information to complete an answer.

5. ANSWER QUALITY

- Provide clear, structured and informative answers.
- Use headings and bullet points when appropriate.
- Explain technical concepts using the document's terminology.
- Avoid unnecessary repetition.
- Keep answers focused on the user's question.

6. SOURCE ATTRIBUTION

- Use the source identifiers supplied in the context.
- Cite supporting passages using [filename, page, chunk].
- Never invent page numbers or source references.

7. COMPARATIVE QUESTIONS

When asked to compare two or more concepts:

- Explain each concept separately.
- Identify their key differences.
- Include examples from the document when available.
- Provide a short concluding comparison.

Your highest priorities are factual accuracy, evidence-based
reasoning and faithful representation of the documents.
"""


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client() -> Groq:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(
        api_key=api_key,
        timeout=120.0,
        max_retries=2
    )


# ============================================================
# CONTEXT CONSTRUCTION
# ============================================================

def build_context(documents):

    context_parts = []

    for index, document in enumerate(documents, start=1):

        filename = document.metadata.get(
            "filename", "document"
        )

        page = document.metadata.get(
            "page", "?"
        )

        chunk = document.metadata.get(
            "chunk_id", "?"
        )

        context_parts.append(
            f"""
SOURCE {index}:
[{filename}, p{page}-c{chunk}]

CONTENT:
{document.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_sources(documents):

    sources = []

    for document in documents:

        sources.append({
            "filename": document.metadata.get(
                "filename", ""
            ),

            "page": document.metadata.get(
                "page", -1
            ),

            "chunk_id": document.metadata.get(
                "chunk_id", -1
            ),

            "text": document.page_content
        })

    return sources


# ============================================================
# ANSWER GENERATION
# ============================================================

def answer_question(doc_ids, question: str):

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # STEP 1: VALIDATE QUESTION
    # --------------------------------------------------------

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    # --------------------------------------------------------
    # STEP 2: RETRIEVE DOCUMENT CHUNKS
    # --------------------------------------------------------

    logger.info("Starting document retrieval.")

    documents = retrieve(
        doc_ids,
        question
    )

    retrieval_time = time.perf_counter() - start_time

    logger.info(
        "Retrieved %d document chunks in %.2f seconds.",
        len(documents),
        retrieval_time
    )

    # --------------------------------------------------------
    # STEP 3: HANDLE EMPTY RETRIEVAL
    # --------------------------------------------------------

    if not documents:

        return (
            "I couldn't find sufficient information about "
            "this in the uploaded documents.",
            []
        )

    # --------------------------------------------------------
    # STEP 4: CONSTRUCT CONTEXT
    # --------------------------------------------------------

    context = build_context(documents)

    logger.info(
        "Context constructed. Characters: %d",
        len(context)
    )

    # --------------------------------------------------------
    # STEP 5: CONSTRUCT USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


USER QUESTION:

{question}


INSTRUCTIONS:

1. Read all retrieved passages carefully.

2. Identify every passage relevant to the question.

3. Combine information from multiple passages when necessary.

4. For comparative questions, explain both concepts,
   their differences and available examples.

5. For multi-part questions, answer every supported part.

6. If partial information is available, provide it and
   clearly identify missing information.

7. Use only the supplied document context.

8. Include source references in your answer.

Generate a clear and informative response.
"""

    # --------------------------------------------------------
    # STEP 6: GENERATE ANSWER USING GROQ
    # --------------------------------------------------------

    logger.info(
        "Generating answer using model: %s",
        GROQ_MODEL_NAME
    )

    generation_start = time.perf_counter()

    completion = get_groq_client().chat.completions.create(

        model=GROQ_MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0,

        max_completion_tokens=MAX_NEW_TOKENS,

        reasoning_effort="low",

        reasoning_format="parsed"
    )

    generation_time = (
        time.perf_counter() - generation_start
    )

    # --------------------------------------------------------
    # STEP 7: EXTRACT AND VALIDATE RESPONSE
    # --------------------------------------------------------

    if not completion.choices:

        raise RuntimeError(
            "Groq returned no completion choices."
        )

    choice = completion.choices[0]

    message = choice.message

    answer = message.content

    # --------------------------------------------------------
    # STEP 8: DEBUG INFORMATION
    # --------------------------------------------------------

    logger.info(
        """
        ========== GROQ RESPONSE DEBUG ==========
        Model: %s
        Finish reason: %s
        Content available: %s
        Generation time: %.2f seconds
        Usage: %s
        =========================================
        """,

        completion.model,

        choice.finish_reason,

        bool(answer),

        generation_time,

        completion.usage
    )

    # --------------------------------------------------------
    # STEP 9: HANDLE EMPTY RESPONSE
    # --------------------------------------------------------

    if not answer or not answer.strip():

        logger.error(
            "Groq returned an empty response. "
            "Finish reason: %s",
            choice.finish_reason
        )

        raise RuntimeError(
            "Groq returned an empty answer. "
            f"Finish reason: {choice.finish_reason}. "
            "Check the Groq response configuration."
        )

    answer = answer.strip()

    # --------------------------------------------------------
    # STEP 10: EXTRACT SOURCE REFERENCES
    # --------------------------------------------------------

    sources = extract_sources(documents)

    # --------------------------------------------------------
    # STEP 11: TOTAL EXECUTION TIME
    # --------------------------------------------------------

    total_time = time.perf_counter() - start_time

    logger.info(
        "Question answered successfully in %.2f seconds.",
        total_time
    )

    # --------------------------------------------------------
    # STEP 12: RETURN RESULT
    # --------------------------------------------------------

    return answer, sources