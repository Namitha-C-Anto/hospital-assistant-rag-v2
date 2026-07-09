def _latency_summary(evaluation_results, num_questions):
    
    return {
        "avg_retrieval_latency" : round(
            sum(
                r.latency.retrieval_seconds
                for r in evaluation_results
            ) / num_questions,
            4
        ),
        "avg_reranker_latency" : round(
            sum(
                r.latency.reranker_seconds
                for r in evaluation_results
            ) / num_questions,
            4
        ),
        "avg_generation_latency": round(
            sum(
                r.latency.generation_seconds
                for r in evaluation_results
            ) / num_questions,
            4
        ),
        "avg_pipeline_latency": round(
            sum(
                r.latency.pipeline_seconds
                for r in evaluation_results
            ) / num_questions,
            4
        ),
        "avg_prompt_latency": round(
            sum(
                r.latency.prompt_seconds
                for r in evaluation_results
            ) / num_questions,
            4
        )
    }

def _token_summary(evaluation_results, num_questions):
    
    total_prompt_tokens = sum(
        r.usage.prompt_tokens
        for r in evaluation_results
    )

    total_completion_tokens = sum(
        r.usage.completion_tokens
        for r in evaluation_results
    )

    total_tokens = sum(
        r.usage.total_tokens
        for r in evaluation_results
    )

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": round(
            total_prompt_tokens / num_questions, 2),
        "avg_completion_tokens": round(
            total_completion_tokens/ num_questions, 2),
        "avg_total_tokens": round(
            total_tokens / num_questions, 2)
    }

def _chunk_summary(evaluation_results, num_questions):
    
    return {
        "avg_chunks_retrieved": round(
            sum(
                r.retrieval_stats.retrieved
                for r in evaluation_results
            ) / num_questions,
            2
        ),
        
        "avg_chunks_after_reranker": round(
            sum(
                r.retrieval_stats.after_reranker
                for r in evaluation_results
            ) / num_questions,
            2
        )
    }

def _ragas_summary(df):
    summary = {}

    for col in [
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
        "context_precision",
        "context_recall",
    ]:
        if col in df.columns:
            summary[col] = round(df[col].mean(), 3)

    return summary

#------------------------------------------------------------------------------

def build_summary(test_data, evaluation_results, ragas_time, df):
    
    if not evaluation_results:
        raise RuntimeError("No successful test cases were collected.")

    num_questions = len(evaluation_results)

    summary = {
        "total_questions": len(test_data),
        "successful_questions": len(evaluation_results),
        "failed_questions": len(test_data) - len(evaluation_results),
    }

    summary.update(_latency_summary(evaluation_results, num_questions))
    summary.update(_chunk_summary(evaluation_results, num_questions))
    summary.update(_token_summary(evaluation_results, num_questions))
    summary.update(_ragas_summary(df))

    summary["ragas_evaluation_seconds"] = ragas_time

    return summary