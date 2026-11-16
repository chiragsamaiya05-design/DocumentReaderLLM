import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from groq import APIStatusError, APIConnectionError
from fastapi.middleware.cors import CORSMiddleware

from app.config import INDEX_DIR, MAX_UPLOAD_BYTES, UPLOAD_DIR
from app.ingestion.loader import load_and_split_pdf
from app.retrieval.vector_store import build_index, index_exists
from app.generation.qa_chain import answer_question
from app.models.schemas import UploadResponse, AskRequest, AskResponse, SourceChunk

app = FastAPI(title="DocuMind (LangChain)", description="Multi-PDF Q&A with source citations, built on LangChain")

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:8501"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        FRONTEND_URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}.pdf"

    save_path.write_bytes(content)

    try:
        chunks = load_and_split_pdf(str(save_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        for chunk in chunks:
            chunk.metadata["filename"] = filename

        build_index(doc_id, chunks)
    except HTTPException:
        save_path.unlink(missing_ok=True)
        shutil.rmtree(INDEX_DIR / doc_id, ignore_errors=True)
        raise
    except Exception as error:
        save_path.unlink(missing_ok=True)
        shutil.rmtree(INDEX_DIR / doc_id, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Could not process this PDF.") from error

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        num_chunks=len(chunks),
        message=f"Document indexed successfully. Use doc_id='{doc_id}' to ask questions.",
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    # Step 1: Get selected document IDs
    doc_ids = request.selected_doc_ids()

    if not doc_ids:
        raise HTTPException(
            status_code=400,
            detail="Select at least one indexed document."
        )

    # Step 2: Check whether all selected documents exist
    missing = [
        doc_id for doc_id in doc_ids
        if not index_exists(doc_id)
    ]

    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown document ID(s): {', '.join(missing)}"
        )

    # Step 3: Generate answer
    try:
        result = answer_question(
            doc_ids,
            request.question,
            include_debug=request.include_debug
        )

        if request.include_debug:
            answer, sources, debug_info = result
        else:
            answer, sources = result
            debug_info = None

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        ) from error

    except APIStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Groq API error: {error.status_code}"
        ) from error

    except APIConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Groq API."
        ) from error

    # Step 4: Check whether relevant sources were retrieved
    if not sources:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found."
        )

    # Step 5: Convert sources into Pydantic models
    source_chunks = [
        SourceChunk(
            filename=source.get("filename", ""),
            page=source["page"],
            chunk_id=source["chunk_id"],
            text=source["text"]
        )
        for source in sources
    ]

    # Step 6: Return answer, sources, and optional debug information
    return AskResponse(
        answer=answer,
        sources=source_chunks,
        debug=debug_info
    )

@app.get("/")
async def root():
    return {"status": "DocuMind (LangChain) API is running"}
