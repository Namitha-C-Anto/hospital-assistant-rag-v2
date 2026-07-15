from evaluate.test_questions import TEST_DATA
from evaluate.rag_runner import run_test_questions
from evaluate.ragas_runner import (
    run_ragas_evaluation, 
    attach_metrics_to_pipeline_results,)
from evaluate.result_writer import (
    add_metadata, 
    save_results,)
from evaluate.display import display_ragas_summary
from evaluate.summary import build_summary
from evaluate.logging import (
    log_df_info, 
    log_avg_scores,)

from rag.initializer import (
    initialize_rag, 
    initialize_ragas,)

from utils.logger import logger
from utils.experiment import create_experiment_metadata

def main():

    """Run the complete RAG evaluation pipeline."""

    logger.info("Starting RAG evaluation...")

    # Initialize
    rag_components = initialize_rag()
    ragas_components = initialize_ragas()
    
    # Run RAG pipeline
    pipeline_results = run_test_questions(
        TEST_DATA, 
        rag_components,
    )

    # Evaluate with RAGAS
    ragas_results, ragas_time = run_ragas_evaluation(
        pipeline_results, 
        ragas_components,
    )

    # Prepare results
    df = attach_metrics_to_pipeline_results(
        ragas_results, 
        pipeline_results,
    )

    metadata = create_experiment_metadata() 
    df = add_metadata(
        df,
        metadata,
    )

    # Display
    display_ragas_summary(df)

    summary = build_summary(
        TEST_DATA, 
        pipeline_results, 
        ragas_time, 
        df,
    ) 

    log_df_info(df)
    log_avg_scores(summary)

    # Save outputs
    save_results(
        metadata, 
        summary, 
        TEST_DATA, 
        pipeline_results,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Evaluation pipeline failed.")
        raise

