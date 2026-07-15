import time

from langchain_core.documents import Document

from config import USE_RERANKER, TOP_K
from rag.models import RetrievalResult, DocumentInfo
from rag.retriever import retrieve_documents
from rag.reranker import reranker

##--------------------DEDUPLICATION
def deduplicate_documents(
    documents: list[Document]
) -> list[Document]:

    """Remove duplicate documents based on source, page and chunk_id."""
         
    seen = set()
    unique_documents = []

    for document in documents:
        
        key = (
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.metadata.get("chunk_id", document.page_content)
        )

        if key not in seen:
            seen.add(key)
            unique_documents.append(document)

    return unique_documents

def build_retrieval_result(
    retrieved_documents: list[Document], 
    reranked_documents: list[Document]
) -> RetrievalResult:
    
    """
    Convert retrieved and reranked LangChain documents into a RetrievalResult.
    """

    return RetrievalResult(
        retrieved_documents=[
            DocumentInfo(
                    content=document.page_content,
                    metadata=document.metadata,
                )
                for document in retrieved_documents
            ],
            reranked_documents=[
                DocumentInfo(
                    content=document.page_content,
                    metadata=document.metadata,
                )
                for document in reranked_documents
            ],
        )

def run_retrieval_pipeline(
    question: str,
    faiss_retriever,
    bm25_retriever
) -> tuple[RetrievalResult, float, float]:

    """
    Retrieve, deduplicate, and optionally rerank documents.

    Args:
        question: User query.
        faiss_retriever: FAISS retriever.
        bm25_retriever: BM25 retriever.

    Returns:
        A tuple containing:
            - RetrievalResult
            - Retrieval latency (seconds)
            - Reranker latency (seconds)
    """
    
    # ----------------------------------------
    # Retrieve documents
    # ---------------------------------------- 
    retrieval_start = time.perf_counter()
    retrieved_documents = retrieve_documents(
                question,
                faiss_retriever,
                bm25_retriever,
            )
    retrieval_time = round(
        time.perf_counter() - retrieval_start, 
        4,
    )

    # ----------------------------------------
    # Remove duplicate documents
    # ----------------------------------------
    unique_documents = deduplicate_documents(retrieved_documents)

    # ----------------------------------------
    # Rerank documents (optional)
    # ----------------------------------------
    reranker_time = 0.0

    # Rerank them
    if USE_RERANKER:
        reranker_start = time.perf_counter()

        reranked_documents  = reranker.compress_documents(
            documents=unique_documents,
            query=question
        )
        reranker_time = round(
            time.perf_counter() - reranker_start,
            4,
        )
    else:
        reranked_documents  = unique_documents[:TOP_K]

    # ----------------------------------------
    # Build retrieval metadata
    # ----------------------------------------
    retrieval_result = build_retrieval_result(
        retrieved_documents=unique_documents,
        reranked_documents=reranked_documents,
    )
 
    return (
        retrieval_result,
        retrieval_time,
        reranker_time,
    )