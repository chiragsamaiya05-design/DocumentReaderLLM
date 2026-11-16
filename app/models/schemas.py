from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str


class AskRequest(BaseModel):
    
    include_debug: bool = False
    question: str
    doc_ids: List[str] = Field(default_factory=list)
    doc_id: Optional[str] = None

    def selected_doc_ids(self) -> List[str]:
        return list(dict.fromkeys(
            self.doc_ids or ([self.doc_id] if self.doc_id else [])
            ))


class SourceChunk(BaseModel):
    filename: str = ""
    page: int
    chunk_id: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    debug: Optional[dict] = None