import os
import time
from config import OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,FETCH_K, LAMBDA_MULT, USE_RERANKER, RERANKER_MODEL, RERANKER_TOP_N, EMBEDDING_MODEL, SEARCH_TYPE, LLM_MODEL, JUDGE_MODEL, DATASET, TEMPERATURE, RETRIEVAL_MODE, EVALUATION_PATH, RAGAS_RESULTS_PATH, DEBUG
from datetime import datetime
from datasets import Dataset
import pandas as pd
from rag.retriever import create_retriever
from evaluate.test_questions import TEST_DATA
from ragas import evaluate 
import json  
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,  # Newer equivalent of ContextPrecisionWithReference
    ContextRecall
)
from utils.logger import logger  
from dataclasses import asdict
from rag.models import ( 
    Latency,
    RetrievalStats, 
    TokenUsage,
    PipelineResult,
    Metrics,
)
from rag.retrieval import run_retrieval_pipeline
from rag.pipeline import initialize_pipeline
from rag.generation import generate_answer

# -------------------------------------------------
# 1. Load RAG components
# -------------------------------------------------

vectorstore, retriever, app_llm, judge_llm, ragas_llm, ragas_embeddings,faiss_retriever, bm25_retriever = initialize_pipeline()

evaluation_results = []

logger.info("Running RAG on test questions...")
 
for item in TEST_DATA:
    
    question = item["question"]
    pipeline_start = time.perf_counter()

    try:  
        
        retrieval_result, retrieval_time, reranker_time = run_retrieval_pipeline(
            question, faiss_retriever, bm25_retriever)
         
        context_text = "\n\n---\n\n".join(
            doc.content
            for doc in retrieval_result.reranked_documents
            )

        answer, usage, prompt_time, generation_time = generate_answer(
            question, context_text, app_llm)

        pipeline_time = round(time.perf_counter() - pipeline_start,4)

        evaluation_results.append(
            PipelineResult(
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
        )
        logger.info(
            f"{question} | "
            f"Retrieval={retrieval_time}s | "
            f"Reranker={reranker_time}s | "
            f"Generation={generation_time}s | "
            f"Pipeline={pipeline_time}s"
        )
            
        if DEBUG:
            logger.info("="*80)
            logger.info("QUESTION")
            logger.info(question)

            logger.info("\nRESPONSE")
            logger.info(answer)

            logger.info("\nREFERENCE")
            logger.info(item["ground_truth"])

            logger.info("\nCONTEXT")
            logger.info(context_text)
            logger.info("="*80)

            for i, ctx in enumerate(retrieval_result.reranked_documents, 1):
                logger.info(f"\nChunk {i}")
                logger.info(ctx.metadata)
                logger.info(ctx.content)
            
    except Exception as e:
        logger.exception(f"Failed question: {question}")
        logger.exception(e)

# -------------------------------------------------
# 3. Build RAGAS dataset
# -------------------------------------------------

ragas_dataset = Dataset.from_dict(
    {
        "user_input": [
            x.question
            for x in evaluation_results
        ],

        "response": [
            x.answer
            for x in evaluation_results
        ],

        "retrieved_contexts": [
            [
                chunk.content
                for chunk in (
                    x.retrieval.reranked_documents
                    if USE_RERANKER
                    else x.retrieval.retrieved_documents
                )
            ]
            for x in evaluation_results
        ],

        "reference": [
            x.reference
            for x in evaluation_results
        ],
    }
)
# -------------------------------------------------
# 4. Configure metrics
# -------------------------------------------------

metrics = [
    Faithfulness(llm=ragas_llm),
    ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm)
]
# -------------------------------------------------
# 5. Run evaluation
# -------------------------------------------------

logger.info("Evaluating with RAGAS...")
if not evaluation_results:
    raise RuntimeError("No successful test cases were collected.")

try:
    ragas_start = time.perf_counter()
    
    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
    )
    ragas_time = round(time.perf_counter() - ragas_start, 4)

except Exception as e:
    logger.exception("RAGAS evaluation failed")
    logger.exception(e)
    raise
# -------------------------------------------------
# 6. Display results
# -------------------------------------------------

logger.info("========== RAGAS SCORES ==========")
logger.info(results)

df = results.to_pandas()
for result, (_, row) in zip(evaluation_results, df.iterrows()):

    result.metrics = Metrics(
        faithfulness= row["faithfulness"],
        answer_relevancy= row["answer_relevancy"],
        context_precision= row["context_precision"],
        context_recall= row["context_recall"],
    )
        
    
RUN_DATE = datetime.now().strftime("%Y%m%d_%H%M%S")

EXPERIMENT_NAME = (
    f"{RETRIEVAL_MODE}_{RUN_DATE}_"
    f"{'reranker' if USE_RERANKER else 'no_reranker'}"
)
logger.info(f"Experiment: {EXPERIMENT_NAME}")

experiment_info = {
    "experiment": EXPERIMENT_NAME,
    "run_date": RUN_DATE,

    "dataset": DATASET,
    "temperature": TEMPERATURE,

    "retrieval_mode": RETRIEVAL_MODE,
    "search_type": SEARCH_TYPE,

    "embedding_model": EMBEDDING_MODEL,

    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,

    "top_k": TOP_K,
    "fetch_k": FETCH_K,
    "lambda_mult": LAMBDA_MULT,

    "use_reranker": USE_RERANKER,
    "reranker_model": RERANKER_MODEL if USE_RERANKER else None,
    "reranker_top_n": RERANKER_TOP_N if USE_RERANKER else None,

    "generation_llm": LLM_MODEL,
    "judge_llm": JUDGE_MODEL,
}

metadata = {
    "experiment": EXPERIMENT_NAME,
    "run_date": RUN_DATE,
    "retrieval_mode": RETRIEVAL_MODE,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "top_k": TOP_K,
    "fetch_k": FETCH_K,
    "lambda_mult": LAMBDA_MULT,
    "search_type": SEARCH_TYPE,
    "use_reranker": USE_RERANKER,
    "reranker_model": RERANKER_MODEL,
    "reranker_top_n": RERANKER_TOP_N,
    "embedding_model": EMBEDDING_MODEL,
    "generation_llm": LLM_MODEL,
    "judge_llm": JUDGE_MODEL,
}

for key, value in metadata.items():
    df[key] = value

logger.info("\nAvailable columns:")
logger.info(df.columns.tolist())

logger.info("\n--- Per Question Breakdown ---")

columns_to_show = [
    col for col in [
        "user_input",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    if col in df.columns
]

logger.info(df[columns_to_show].to_string(index=False))
# -------------------------------------------------Display

summary = {
    "total_questions": len(TEST_DATA),
    "successful_questions": len(evaluation_results),
    "failed_questions": len(TEST_DATA) - len(evaluation_results),
}

num_questions = max(len(evaluation_results), 1)

summary["avg_retrieval_latency"] = round(
    sum(
        r.latency.retrieval_seconds
        for r in evaluation_results
    ) / num_questions,
    4
)

summary["avg_reranker_latency"] = round(
    sum(
        r.latency.reranker_seconds
        for r in evaluation_results
    ) / num_questions,
    4
)

summary["avg_generation_latency"] = round(
    sum(
        r.latency.generation_seconds
        for r in evaluation_results
    ) / num_questions,
    4
)

summary["avg_pipeline_latency"] = round(
    sum(
        r.latency.pipeline_seconds
        for r in evaluation_results
    ) / num_questions,
    4
)

summary["avg_prompt_latency"] = round(
    sum(
        r.latency.prompt_seconds
        for r in evaluation_results
    ) / num_questions,
    4
)

summary["total_prompt_tokens"] = sum(
    r.usage.prompt_tokens
    for r in evaluation_results
)

summary["total_completion_tokens"] = sum(
    r.usage.completion_tokens
    for r in evaluation_results
)

summary["total_tokens"] = sum(
    r.usage.total_tokens
    for r in evaluation_results
)
summary["avg_prompt_tokens"] = round(
    summary["total_prompt_tokens"] / num_questions, 2
)

summary["avg_completion_tokens"] = round(
    summary["total_completion_tokens"] / num_questions, 2
)

summary["avg_total_tokens"] = round(
    summary["total_tokens"] / num_questions, 2
)
summary["avg_chunks_retrieved"] = round(
    sum(
        r.retrieval_stats.retrieved
        for r in evaluation_results
    ) / num_questions,
    2
)

summary["avg_chunks_after_reranker"] = round(
    sum(
        r.retrieval_stats.after_reranker
        for r in evaluation_results
    ) / num_questions,
    2
)

summary["ragas_evaluation_seconds"] = ragas_time

for col in [
    "faithfulness",
    "answer_relevancy",
    "response_relevancy",
    "context_precision",
    "context_recall"
]:
    if col in df.columns:
        summary[col] = round(df[col].mean(), 3)

logger.info(df[[
    "user_input",
    "response",
    "reference",
    "retrieved_contexts",
    "faithfulness"
]])

summary_row = {
    **metadata,
    "dataset": DATASET,
    "temperature": TEMPERATURE,

    "faithfulness": summary.get("faithfulness"),
    "answer_relevancy": summary.get("answer_relevancy"),
    "context_precision": summary.get("context_precision"),
    "context_recall": summary.get("context_recall"),
    "avg_pipeline_latency": summary.get("avg_pipeline_latency"),
    "avg_generation_latency": summary.get("avg_generation_latency"),
    "avg_retrieval_latency": summary.get("avg_retrieval_latency"), 
    "avg_prompt_latency": summary.get("avg_prompt_latency"),
    "avg_total_tokens": summary.get("avg_total_tokens"),
    "ragas_evaluation_seconds": summary.get("ragas_evaluation_seconds"),
    "avg_reranker_latency": summary.get("avg_reranker_latency"),
    "avg_chunks_retrieved": summary.get("avg_chunks_retrieved"),
    "avg_chunks_after_reranker": summary.get("avg_chunks_after_reranker"),
    "avg_prompt_tokens": summary.get("avg_prompt_tokens"),
    "avg_completion_tokens": summary.get("avg_completion_tokens"), 
}
 
summary_path = os.path.join(EVALUATION_PATH, "experiment_summary.csv")
summary_df = pd.DataFrame([summary_row])

if os.path.exists(summary_path):

    existing_df = pd.read_csv(summary_path)

    summary_df = pd.concat(
        [existing_df, summary_df],
        ignore_index=True
    )

summary_df.to_csv(summary_path, index=False)
#---------------------------------------------

logger.info("\n========== AVERAGE SCORES ==========")
for metric, score in summary.items():
    logger.info(f"{metric}: {score}")

# -------------------------------------------------
# 7. Save results
# -------------------------------------------------

output_dir = RAGAS_RESULTS_PATH
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, f"{EXPERIMENT_NAME}.json")

json_data = {
    "version": "1.0",
    "experiment": experiment_info,
    "summary": summary, 
    "run_statistics": {
        "total_questions": len(TEST_DATA),
        "successful_questions": len(evaluation_results),
        "failed_questions": len(TEST_DATA) - len(evaluation_results),
    },

    "results": [
        asdict(result)
        for result in evaluation_results
    ]
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        json_data,
        f,
        indent=4,
        ensure_ascii=False,
    )

logger.info(f"✅ JSON saved to {output_path}")
