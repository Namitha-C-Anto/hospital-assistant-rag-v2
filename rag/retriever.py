from config import (
    SEARCH_TYPE,
    TOP_K,
    FETCH_K,
    LAMBDA_MULT,
    CHUNKS_PATH,
    RETRIEVAL_MODE,
)

from langchain_community.retrievers import BM25Retriever
from rag.storage import load_chunks
from rag.rrf import reciprocal_rank_fusion


def create_retriever(vectorstore):

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

    if RETRIEVAL_MODE == "faiss":
        return faiss_retriever

    documents = load_chunks(CHUNKS_PATH)

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = TOP_K

    return {
        "faiss": faiss_retriever,
        "bm25": bm25_retriever,
    }

##-----------------------------------------------------------------------------------------

def retrieve_documents(question, faiss_retriever, bm25_retriever=None):
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