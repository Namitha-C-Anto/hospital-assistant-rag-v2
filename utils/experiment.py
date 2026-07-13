from typing import Any
from config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    TOP_K,
    FETCH_K, 
    LAMBDA_MULT, 
    USE_RERANKER, 
    RERANKER_MODEL, 
    RERANKER_TOP_N, 
    EMBEDDING_MODEL, 
    SEARCH_TYPE, 
    LLM_MODEL, 
    JUDGE_MODEL, 
    DATASET, 
    TEMPERATURE, 
    RETRIEVAL_MODE, 
    RUN_DATE,)
from utils.logger import logger


def create_experiment_metadata() -> dict[str, Any]:
    """
    Create metadata describing the current RAG experiment.

    The metadata captures the retrieval, reranking, embedding,
    and LLM configuration used for the experiment. It is stored
    alongside the evaluation results to make experiments
    reproducible and easy to compare.

    Returns:
        Dictionary containing the experiment configuration.
    """

    # -------------------------------------------------
    # Generate a descriptive experiment name
    # -------------------------------------------------
    
    experiment_name = (
        f"{RETRIEVAL_MODE}_{RUN_DATE}_"
        f"{'reranker' if USE_RERANKER else 'no_reranker'}"
    )
    logger.info(f"Experiment: {experiment_name}")

    # -------------------------------------------------
    # Record the configuration used for this experiment
    # -------------------------------------------------
    metadata = {
        "experiment": experiment_name,
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
 
    return metadata