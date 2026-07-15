from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from config import JUDGE_MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL
from rag.vectorstore import load_vectorstore
from rag.retriever import create_retriever
from rag.models import PipelineComponents, EvaluationComponents
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper 

def initialize_rag() -> PipelineComponents:
    """
    Initialize and return all components required for the RAG pipeline.

    This includes:
    - Loading the persisted vector store
    - Creating retrieval components (FAISS + BM25)
    - Initializing the application LLM 

    Returns:
        PipelineComponents: Container holding all initialized pipeline objects.
    """

    # -------------------------------------------------
    # 1. Load vector store and create retrievers
    # -------------------------------------------------
    vectorstore = load_vectorstore()

    retriever = create_retriever(vectorstore)

    # Individual retrievers available for experimentation and retrieval strategies
    faiss_retriever = retriever["faiss"]
    bm25_retriever = retriever["bm25"]

    # -------------------------------------------------
    # 2. Initialize the application LLM used for answer generation
    # -------------------------------------------------
    app_llm = ChatOpenAI(
        model= LLM_MODEL,
        api_key= OPENAI_API_KEY,
        )

    # -------------------------------------------------
    # 3. Return all initialized components
    # -------------------------------------------------
    return PipelineComponents(
        vectorstore=vectorstore,
        retriever=retriever,
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        app_llm=app_llm,
    )
    

def initialize_ragas() -> EvaluationComponents:
    """
    Initialize and return all components required for the RAGAS Evaluation.

    This includes:
    - Initializing the evaluation (RAGAS) LLM and embedding model

    Returns:
        EvaluationComponents: Container holding all initialized ragas evaluation objects.
    """

    # -------------------------------------------------
    # 1. Initialize LLM used as the RAGAS evaluation judge
    # -------------------------------------------------
    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=OPENAI_API_KEY,
    )

    # -------------------------------------------------
    # 2. Initialize embedding model for RAGAS metrics
    # -------------------------------------------------
    ragas_llm = LangchainLLMWrapper(judge_llm) 

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        hf_embeddings
    )

    # -------------------------------------------------
    # 3. Return all initialized components
    # -------------------------------------------------
    return EvaluationComponents(
        judge_llm=judge_llm,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )
    
