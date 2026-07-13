from typing import Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from config import (
    SEARCH_TYPE,
    TOP_K,
    FETCH_K,
    LAMBDA_MULT,
    CHUNKS_PATH,
    RETRIEVAL_MODE,
)
from rag.storage import load_chunks
from rag.rrf import reciprocal_rank_fusion
 

def create_retriever(
    vectorstore,
) -> dict[str, Optional[BaseRetriever]]:
    """
    Create retrievers based on the configured retrieval mode.

    Returns:
        Dictionary containing:
            - "faiss": FAISS retriever
            - "bm25": BM25 retriever or None
    """

    if RETRIEVAL_MODE not in {"faiss", "hybrid"}:
        raise ValueError(f"Unknown retrieval mode: {RETRIEVAL_MODE}")

    faiss_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K,
            "lambda_mult": LAMBDA_MULT,
        },
    )

    bm25_retriever = None
    
    if RETRIEVAL_MODE == "hybrid":
        documents = load_chunks(CHUNKS_PATH)

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = TOP_K

    return {
        "faiss": faiss_retriever,
        "bm25": bm25_retriever,
    }

##-----------------------------------------------------------------------------------------

def retrieve_documents(
    question: str,
    faiss_retriever: BaseRetriever,
    bm25_retriever: Optional[BaseRetriever] = None,
) -> list[Document]:
    """
    Retrieve documents using the configured retrieval mode.

    - FAISS mode: returns vector search results.
    - Hybrid mode: combines FAISS and BM25 using Reciprocal Rank Fusion (RRF).

    Args:
        question: User query.
        faiss_retriever: FAISS retriever.
        bm25_retriever: BM25 retriever (required for hybrid mode).

    Returns:
        List of retrieved documents.
    """

    question = question.strip()

    if RETRIEVAL_MODE == "faiss":
        return faiss_retriever.invoke(question)

    if bm25_retriever is None:
        raise ValueError(
            "Hybrid retrieval requires a BM25 retriever."
        )
        
    faiss_docs = faiss_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    return reciprocal_rank_fusion(
        [faiss_docs, bm25_docs],
        top_n=TOP_K,
    )