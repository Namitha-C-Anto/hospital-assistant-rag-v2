from rag.vectorstore import load_vectorstore
from rag.retriever import create_retriever
from langchain_openai import ChatOpenAI
from config import JUDGE_MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper 

def initialize_pipeline():
    
    vectorstore = load_vectorstore()

    retriever = create_retriever(vectorstore)

    faiss_retriever = retriever["faiss"]
    bm25_retriever = retriever["bm25"]

    #JUDGE_MODEL = LLM_MODEL
    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=OPENAI_API_KEY,
    )

    ragas_llm = LangchainLLMWrapper(judge_llm) 

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        hf_embeddings
    )
    # -------------------------------------------------
    # 2. Run RAG pipeline on test questions
    # -------------------------------------------------
    app_llm = ChatOpenAI(
        model= LLM_MODEL,
        api_key= OPENAI_API_KEY,
        )

    return {
        "vectorstore": vectorstore,
        "retriever": retriever,
        "app_llm": app_llm,
        "judge_llm": judge_llm,
        "ragas_llm": ragas_llm,
        "ragas_embeddings": ragas_embeddings,
        "faiss_retriever": faiss_retriever,
        "bm25_retriever": bm25_retriever
    }
