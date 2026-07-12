import os
from dotenv import load_dotenv
from datetime import datetime

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv(override=True)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.getenv("DATASET", "dataset_v1")

DOCS_PATH = os.path.join(BASE_DIR, "docs", DATASET)
DB_PATH = os.path.join(BASE_DIR, "db", "faiss_index")
CHUNKS_PATH = os.path.join(BASE_DIR, "db", "chunks")

EVALUATION_PATH = os.path.join(BASE_DIR, "evaluate", "evaluation_results")
RAGAS_RESULTS_PATH = os.path.join(EVALUATION_PATH, "ragas_results", DATASET)
COMPARISON_PATH = os.path.join(EVALUATION_PATH, "comparisons")
REPORT_PATH = os.path.join(EVALUATION_PATH, "reports")
DATASET_PATH = os.path.join(EVALUATION_PATH, "datasets")

# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# --------------------------------------------------
# Text Splitting
# --------------------------------------------------

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# --------------------------------------------------
# Retriever Configuration
# --------------------------------------------------
# --------------------------------------------------
# Retriever Configuration
# --------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 5))
FETCH_K = int(os.getenv("FETCH_K", 20))

SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")

LAMBDA_MULT = float(os.getenv("LAMBDA_MULT", 0.5))
# --------------------------------------------------
# Reranker Configuration
# --------------------------------------------------

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)

RERANKER_TOP_N = int(
    os.getenv("RERANKER_TOP_N", 3)
)

#--------------------------------------------------Toggle
USE_RERANKER = os.getenv(
    "USE_RERANKER",
    "True"
).lower() == "true"

RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "faiss").lower()

DEBUG = os.getenv("DEBUG", "false").lower() == "false"
# --------------------------------------------------
# Large Language Model
# --------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
# --------------------------------------------------
# Evaluation LLM
# --------------------------------------------------

JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    LLM_MODEL
)
# --------------------------------------------------
# Streamlit Application
# --------------------------------------------------

APP_TITLE = "🏥 Conversational Hospital Assistant"

SUB_TITLE = (
    "AI-powered hospital knowledge assistant "
    "with memory and retrieval capabilities"
)

# --------------------------------------------------
# API Keys
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# Validation
# --------------------------------------------------

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Please add it to your .env file."
    )

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found.")

RUN_DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
