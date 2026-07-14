"""
Uses LangChain's PyPDFLoader (loads each page as a Document with
page_content + metadata['page']) and RecursiveCharacterTextSplitter
to produce citable chunks. LangChain preserves page metadata across
the split automatically, which is what we relied on manually before.
"""
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split_pdf(filepath: str) -> List[Document]:
    loader = PyPDFLoader(filepath)
    pages = loader.load()  # each Document has metadata={"source":..., "page": n}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    return chunks
