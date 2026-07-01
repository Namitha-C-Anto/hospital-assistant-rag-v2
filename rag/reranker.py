from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base"
)

reranker = CrossEncoderReranker(
    model=model,
    top_n=3
)