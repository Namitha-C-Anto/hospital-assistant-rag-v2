import os
from dotenv import load_dotenv

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv(override=True)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.path.join(BASE_DIR, "docs")
DB_PATH = os.path.join(BASE_DIR, "db", "faiss_index")
EVALUATION_PATH = os.path.join(BASE_DIR, "evaluate")

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

TOP_K = int(os.getenv("TOP_K", 5))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")

# --------------------------------------------------
# Large Language Model
# --------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))

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