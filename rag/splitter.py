from typing import Sequence
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def split_documents(
    documents: Sequence[Document],
) -> list[Document]:
    """
    Split documents into overlapping chunks for indexing.

    Args:
        documents: Collection of loaded documents.

    Returns:
        List of chunked documents.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return text_splitter.split_documents(documents)