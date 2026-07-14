import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.config import UPLOAD_DIR
from app.ingestion.loader import load_and_split_pdf
from app.retrieval.vector_store import build_index, index_exists
from app.generation.qa_chain import answer_question
from app.models.schemas import UploadResponse, AskRequest, AskResponse, SourceChunk

app = FastAPI(title="DocuMind (LangChain)", description="Multi-PDF Q&A with source citations, built on LangChain")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}.pdf"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # --- LangChain ingestion pipeline ---
    chunks = load_and_split_pdf(str(save_path))
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    build_index(doc_id, chunks)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=len(chunks),
        message=f"Document indexed successfully. Use doc_id='{doc_id}' to ask questions.",
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if not index_exists(request.doc_id):
        raise HTTPException(status_code=404, detail="doc_id not found. Upload the document first.")

    try:
        answer, sources = answer_question(request.doc_id, request.question)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    if not sources:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    source_chunks = [SourceChunk(page=s["page"], text=s["text"]) for s in sources]
    return AskResponse(answer=answer, sources=source_chunks)


@app.get("/")
async def root():
    return {"status": "DocuMind (LangChain) API is running"}
