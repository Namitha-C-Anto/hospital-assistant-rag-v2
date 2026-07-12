from rag.vectorstore import load_vectorstore
from rag.retriever import create_retriever
from langchain_openai import ChatOpenAI
from config import JUDGE_MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper 
from rag.models import PipelineComponents

def initialize_pipeline() -> PipelineComponents:
    """
    Initialize and return all components required for the RAG pipeline.

    This includes:
    - Loading the persisted vector store
    - Creating retrieval components (FAISS + BM25)
    - Initializing the application LLM
    - Initializing the evaluation (RAGAS) LLM and embedding model

    Returns:
        PipelineComponents: Container holding all initialized pipeline objects.
    """

    # -------------------------------------------------
    # 1. Load vector store and create retrievers
    # -------------------------------------------------
    vectorstore = load_vectorstore()

    retriever = create_retriever(vectorstore)

    # Individual retrievers used for evaluation experiments
    faiss_retriever = retriever["faiss"]
    bm25_retriever = retriever["bm25"]

    # -------------------------------------------------
    # 2. Initialize LLM used as the RAGAS evaluation judge
    # -------------------------------------------------
    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=OPENAI_API_KEY,
    )

    # -------------------------------------------------
    # 3. Initialize embedding model for RAGAS metrics
    # -------------------------------------------------
    ragas_llm = LangchainLLMWrapper(judge_llm) 

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        hf_embeddings
    )
    # -------------------------------------------------
    # 4. Initialize the application LLM used for answer generation
    # -------------------------------------------------
    app_llm = ChatOpenAI(
        model= LLM_MODEL,
        api_key= OPENAI_API_KEY,
        )

    # -------------------------------------------------
    # 5. Return all initialized components
    # -------------------------------------------------
    return PipelineComponents(
        vectorstore=vectorstore,
        retriever=retriever,
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        app_llm=app_llm,
        judge_llm=judge_llm,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )
    
