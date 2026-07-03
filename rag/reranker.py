from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import RERANKER_TOP_N

model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base"
)

reranker = CrossEncoderReranker(
    model=model,
    top_n=RERANKER_TOP_N
)