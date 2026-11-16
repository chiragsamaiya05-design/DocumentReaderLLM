from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split_pdf(filepath: str) -> List[Document]:
    loader = PyPDFLoader(filepath)
    pages = loader.load()  

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk_number, chunk in enumerate(chunks, start=1):
        
        chunk.metadata["page"] = int(chunk.metadata.get("page", 0)) + 1
        chunk.metadata["chunk_id"] = chunk_number
    return chunks
