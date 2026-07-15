import pickle
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

def save_chunks(
    documents: Sequence[Document], 
    path:str | Path,
) -> None:
    
    """
    Save document chunks to disk using pickle.

    Args:
        documents: Collection of document chunks.
        path: Output file path.
    """
    with open(path, "wb") as file:
        pickle.dump(documents, file)

def load_chunks(
    path: str | Path,
) -> list[Document]:
    """
    Load document chunks from a pickle file.

    Args:
        path: Input file path.

    Returns:
        List of document chunks.
    """

    with open(path, "rb") as file:
        return pickle.load(file)