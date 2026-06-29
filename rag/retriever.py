from config import SEARCH_TYPE, TOP_K

def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        #search_kwargs={"k": TOP_K}
        search_kwargs={
            "k": TOP_K,
            "fetch_k": 10,
            "lambda_mult": 0.9
            }
    )