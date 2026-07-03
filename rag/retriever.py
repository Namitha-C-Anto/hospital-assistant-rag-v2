from config import SEARCH_TYPE, TOP_K, FETCH_K, LAMBDA_MULT, CHUNKS_PATH
from langchain_community.retrievers import BM25Retriever
from rag.storage import load_chunks

def create_retriever(vectorstore):
    
    faiss_retriever = vectorstore.as_retriever(
            search_type=SEARCH_TYPE, 
            search_kwargs={
                "k": TOP_K,
                "fetch_k": FETCH_K,
                "lambda_mult": LAMBDA_MULT
                }
        )
    documents = load_chunks(CHUNKS_PATH)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = TOP_K

    return {
    "faiss": faiss_retriever,
    "bm25": bm25_retriever,
}