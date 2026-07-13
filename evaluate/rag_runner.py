import time
from typing import Any 

from config import DEBUG
from utils.logger import logger
from evaluate.logging import (
    log_pipeline_stats, 
    log_debug_info,)
from rag.generation import generate_answer
from rag.models import (
    PipelineResults, 
    Latency, 
    RetrievalStats, 
    TokenUsage, 
    PipelineComponents,)
from rag.retrieval import run_retrieval_pipeline

def run_test_questions(
    test_data: list[dict[str, Any]],
    rag_components: PipelineComponents,
) -> list[PipelineResults]:

    """Run the RAG pipeline for all test questions.

    Args:
        test_data: List of test questions and reference answers.
        rag_components: Initialized RAG pipeline components.

    Returns:
        List of evaluation results for successfully processed questions.
    """
    pipeline_results = []

    logger.info("Running RAG on test questions...")

    # Execute the RAG pipeline for each test question.
    for item in test_data:
        result = process_test_questions(item, rag_components)

        # Skip questions that failed during pipeline execution.
        if result is not None:
            pipeline_results.append(result)

    return pipeline_results

#------------------------------------------------------
def process_test_questions(
    item: dict[str, Any],
    rag_components: PipelineComponents,
) -> PipelineResults | None:

    """
    Execute the complete RAG pipeline for a single test question.

    This includes:
    - Document retrieval
    - Reranking
    - Answer generation
    - Latency measurement
    - Token usage tracking

    Args:
        item: Test question containing the input question and ground truth.
        rag_components: Initialized RAG pipeline components.

    Returns:
        PipelineResult containing the generated answer and evaluation metadata,
        or None if the pipeline execution fails.
    """

    question = item["question"]
    # Start timer to measure end-to-end pipeline latency.
    pipeline_start = time.perf_counter()

    try:

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
            doc.content
            for doc in retrieval_result.reranked_documents
            )

        # -------------------------------------------------
        # 2. Generate answer using the application LLM
        # -------------------------------------------------    
        answer, usage, prompt_time, generation_time = generate_answer(
                question,
                context_text,
                rag_components.app_llm,
            )

        # Calculate total pipeline execution time.
        pipeline_time = round(time.perf_counter() - pipeline_start,4)

        # -------------------------------------------------
        # 3. Build structured pipeline result
        # -------------------------------------------------
        result = PipelineResults(
                    question=question,
                    answer=answer,
                    reference=item["ground_truth"],
                    ground_truth = item["ground_truth"],
                    
                    latency=Latency(
                        retrieval_seconds=retrieval_time,
                        reranker_seconds=reranker_time,
                        generation_seconds=generation_time,
                        pipeline_seconds=pipeline_time,
                        prompt_seconds=prompt_time,
                    ),

                    retrieval_stats=RetrievalStats(
                        retrieved=len(retrieval_result.retrieved_documents),
                        after_reranker=len(retrieval_result.reranked_documents)
                    ),

                    retrieval=retrieval_result, 

                    usage=TokenUsage(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    ), 
                    metrics = None,
                )

        # Log pipeline performance metrics.
        log_pipeline_stats(
            question,
            retrieval_time,
            reranker_time,
            generation_time,
            pipeline_time,
        )

        # Log detailed debugging information when debug mode is enabled.
        if DEBUG:
            log_debug_info(
                question,
                answer,
                item["ground_truth"],
                context_text,
                retrieval_result,
            )
        return result

    except Exception:
        logger.exception(f"Failed question: {question}")
        return None