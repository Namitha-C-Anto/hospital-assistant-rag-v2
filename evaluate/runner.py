import time
from utils.logger import logger
from rag.retrieval import run_retrieval_pipeline
from rag.generation import generate_answer
from rag.models import PipelineResult, Latency, RetrievalStats, TokenUsage, PipelineComponents
from evaluate.logging import log_pipeline_stats, log_debug_info
from config import DEBUG
from typing import Any

def run_test_questions(
    test_data: list[dict[str, Any]],
    components: PipelineComponents,
) -> list[PipelineResult]:

    """Run the RAG pipeline for all test questions.

    Args:
        test_data: List of test questions and reference answers.
        components: Initialized RAG pipeline components.

    Returns:
        List of evaluation results for successfully processed questions.
    """
    evaluation_results = []

    logger.info("Running RAG on test questions...")

    # Execute the RAG pipeline for each test question.
    for item in test_data:
        result = run_single_question(item, components)

        # Skip questions that failed during pipeline execution.
        if result is not None:
            evaluation_results.append(result)

    return evaluation_results

#------------------------------------------------------
def run_single_question(
    item: dict[str, Any],
    components: PipelineComponents,
) -> PipelineResult | None:

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
        components: Initialized RAG pipeline components.

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
                components.faiss_retriever,
                components.bm25_retriever,
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
                components.app_llm,
            )

        # Calculate total pipeline execution time.
        pipeline_time = round(time.perf_counter() - pipeline_start,4)

        # -------------------------------------------------
        # 3. Build structured pipeline result
        # -------------------------------------------------
        result = PipelineResult(
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