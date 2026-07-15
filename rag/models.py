from dataclasses import dataclass, field
from typing import Any
from langchain_openai import ChatOpenAI

"""
Shared dataclasses used throughout the RAG application and
evaluation pipeline.
"""

@dataclass
class DocumentInfo:
    content: str
    metadata: dict[str, Any]

@dataclass
class RetrievalResult:
    retrieved_documents: list[DocumentInfo]
    reranked_documents: list[DocumentInfo]

@dataclass
class RetrievalStats:
    retrieved: int = 0
    after_reranker: int = 0

@dataclass
class Latency:
    retrieval_seconds: float = 0
    reranker_seconds: float = 0
    prompt_seconds: float = 0
    generation_seconds: float = 0
    pipeline_seconds: float = 0

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Metrics:

    faithfulness: float = 0

    answer_relevancy: float = 0

    context_precision: float = 0

    context_recall: float = 0
 
@dataclass
class PipelineComponents:
    vectorstore: object
    retriever: dict
    faiss_retriever: object
    bm25_retriever: object
    app_llm: ChatOpenAI

@dataclass
class EvaluationComponents:
    judge_llm: ChatOpenAI
    ragas_llm: object
    ragas_embeddings: object
   
@dataclass
class RagPipelineResult:
    answer: str
    usage: dict[str,int]
    retrieval_result: RetrievalResult
    retrieval_time: float
    reranker_time: float
    prompt_time: float
    generation_time: float
    context: str

@dataclass
class PipelineResults:
    question: str
    answer: str = ""
    reference: str = ""
    ground_truth: str = ""
    latency: Latency = field(default_factory=Latency)
    usage: TokenUsage = field(default_factory=TokenUsage)
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    retrieval_stats: RetrievalStats = field(default_factory=RetrievalStats)
    metrics: Metrics | None = None
 