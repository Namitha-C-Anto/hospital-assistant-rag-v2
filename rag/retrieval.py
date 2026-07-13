from rag.models import RetrievalResult, DocumentInfo
from rag.retriever import retrieve_documents
from rag.reranker import reranker
from config import USE_RERANKER, TOP_K
import time

##--------------------DEDUPLICATION
def deduplicate_documents(docs):
    """Remove duplicate documents based on source, page and chunk_id."""
         
    seen = set()
    unique_docs = []

    for doc in docs:
        
        key = (
            doc.metadata["source"],
            doc.metadata["page"],
            doc.metadata.get("chunk_id", doc.page_content)
        )

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs

def build_retrieval_result(docs, reranked_results):
    
    """Convert LangChain Documents into RetrievalResult."""

    return RetrievalResult(
        retrieved_documents=[
            DocumentInfo(
                    content=doc.page_content,
                    metadata=doc.metadata,
                )
                for doc in docs
            ],
            reranked_documents=[
                DocumentInfo(
                    content=doc.page_content,
                    metadata=doc.metadata,
                )
                for doc in reranked_results
            ],
        )

def run_retrieval_pipeline(
    question,
    faiss_retriever,
    bm25_retriever
):
 
    retrieval_start = time.perf_counter()
    docs = retrieve_documents(
                question,
                faiss_retriever,
                bm25_retriever,
            )
    retrieval_time = round(time.perf_counter() - retrieval_start, 4)

    docs = deduplicate_documents(docs)

    #------------------RERANK
        
    reranker_time = 0
    # Rerank them
    if USE_RERANKER:
        reranker_start = time.perf_counter()

        reranked_results = reranker.compress_documents(
            documents=docs,
            query=question
        )
        reranker_time = round(time.perf_counter() - reranker_start,4)
    else:
        reranked_results = docs[:TOP_K]

    retrieved_contexts = [
        doc.page_content for doc in reranked_results
    ]

    retrieval_result = build_retrieval_result(
            docs,
            reranked_results
        )
 
    return (
        retrieval_result,
        retrieval_time,
        reranker_time,
    )