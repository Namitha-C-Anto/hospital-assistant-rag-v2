from rag.models import PipelineComponents, RagPipelineResult
from rag.retrieval import run_retrieval_pipeline
from rag.generation import generate_answer

def run_rag_pipeline(
    question: str,
    rag_components: PipelineComponents,
    chat_history: str = "",
) -> RagPipelineResult:

    """
    Execute the complete RAG pipeline.

    This includes:
    - Document retrieval
    - Deduplication
    - Reranking
    - Context construction
    - Answer generation

    Args:
        question: User question.
        rag_components: Initialized RAG components.

    Returns:
        RagPipelineResult containing the generated answer,
        retrieved documents, latency measurements, and token usage.
    """
    # -------------------------------------------------
    # 1. Retrieve and rerank relevant documents
    # -------------------------------------------------
    retrieval_result, retrieval_time, reranker_time = run_retrieval_pipeline(
            question,
            rag_components.faiss_retriever,
            rag_components.bm25_retriever, 
        )

    # Combine retrieved contexts into a single prompt for the LLM.
    context_text = "\n\n---\n\n".join(
        document.content
        for document in retrieval_result.reranked_documents
        )

    # -------------------------------------------------
    # 2. Generate answer using the application LLM
    # -------------------------------------------------    
    answer, usage, prompt_time, generation_time = generate_answer(
            question,
            context_text,
            rag_components.app_llm,
            chat_history,
        )

    return RagPipelineResult(
        answer=answer,
        context=context_text,
        retrieval_result=retrieval_result,
        usage=usage,
        retrieval_time=retrieval_time,
        reranker_time=reranker_time,
        prompt_time=prompt_time,
        generation_time=generation_time,
    )