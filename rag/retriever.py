from config import SEARCH_TYPE, TOP_K, FETCH_K, LAMBDA_MULT

def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type=SEARCH_TYPE, 
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K,
            "lambda_mult": LAMBDA_MULT
            }
    )