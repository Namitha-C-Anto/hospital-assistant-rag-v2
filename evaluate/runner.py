import time
from utils.logger import logger
from rag.retrieval import run_retrieval_pipeline
from rag.generation import generate_answer
from rag.models import PipelineResult, Latency, RetrievalStats, TokenUsage
from evaluate.logging import log_pipeline_stats, log_debug_info
from config import DEBUG

def run_test_questions(test_data, components):
    evaluation_results = []

    logger.info("Running RAG on test questions...")

    for item in test_data:
        result = run_single_question(item, components)

        if result is not None:
            evaluation_results.append(result)

    return evaluation_results

#------------------------------------------------------
def run_single_question(item, components):

    question = item["question"]
    pipeline_start = time.perf_counter()

    try:

        retrieval_result, retrieval_time, reranker_time = run_retrieval_pipeline(
                question,
                components.faiss_retriever,
                components.bm25_retriever,
            )

        context_text = "\n\n---\n\n".join(
            doc.content
            for doc in retrieval_result.reranked_documents
            )
            
        answer, usage, prompt_time, generation_time = generate_answer(
                question,
                context_text,
                components.app_llm,
            )

        pipeline_time = round(time.perf_counter() - pipeline_start,4)

        result = PipelineResult(
                    question=question,
                    answer=answer,
                    reference=item["ground_truth"],

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
                )

        log_pipeline_stats(
            question,
            retrieval_time,
            reranker_time,
            generation_time,
            pipeline_time,
        )

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