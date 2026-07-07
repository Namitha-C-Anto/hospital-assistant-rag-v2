from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class DocumentInfo:
    content: str
    metadata: Dict[str, Any]

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
class RetrievalStats:
    retrieved: int
    after_reranker: int

@dataclass
class RetrievedContext:
    retrieved_documents: List[DocumentInfo] = field(default_factory=list)
    reranked_documents: List[DocumentInfo] = field(default_factory=list)

@dataclass
class PipelineResult:
    question: str
    answer: str = ""
    reference: str = ""

    latency: Latency = field(default_factory=Latency)
    usage: TokenUsage = field(default_factory=TokenUsage)
    retrieval: RetrievedContext = field(default_factory=RetrievedContext)
    retrieval_stats: RetrievalStats = field(default_factory=RetrievalStats)

@dataclass
class Metrics:

    faithfulness: float = 0

    answer_relevancy: float = 0

    context_precision: float = 0

    context_recall: float = 0
    
@dataclass
class PipelineRequest:
    question: str
    chat_history: str = "",
    metrics: Metrics = field(default_factory=Metrics)
