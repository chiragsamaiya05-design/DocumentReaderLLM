from pydantic import BaseModel
from typing import List


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str


class AskRequest(BaseModel):
    question: str
    doc_id: str


class SourceChunk(BaseModel):
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
