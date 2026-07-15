from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import RERANKER_TOP_N, RERANKER_MODEL

# -------------------------------------------------
# Initialize the cross-encoder model used to
# rerank retrieved documents.
# -------------------------------------------------
model = HuggingFaceCrossEncoder(
    model_name=RERANKER_MODEL
)

# -------------------------------------------------
# Create the reranker that returns the top-N
# most relevant documents.
# -------------------------------------------------
reranker = CrossEncoderReranker(
    model=model,
    top_n=RERANKER_TOP_N
)