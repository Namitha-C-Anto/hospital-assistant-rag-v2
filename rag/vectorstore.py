from typing import Sequence
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    EMBEDDING_MODEL, 
    DB_PATH,)
from utils.logger import logger

#Create embedding model
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create and return the configured embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


#Create Vector Store
def create_vectorstore(documents: Sequence[Document]) -> FAISS:
    """
    Create and return a FAISS vector store from the provided documents.
    """
    embeddings = get_embeddings()
    return FAISS.from_documents(documents=documents, embedding=embeddings)
     

#Save Vector Store
def save_vectorstore(
    vectorstore: FAISS, 
    path: str = DB_PATH
) -> None:
    """
    Save the FAISS vector database to disk.
    """
    vectorstore.save_local(path)


def load_vectorstore(path: str = DB_PATH) -> FAISS:
    """
    Load and return the FAISS vector database from disk.
    """

    logger.info(f"Loading vector database from '{path}'.")

    embeddings = get_embeddings()
    return FAISS.load_local(
        path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )