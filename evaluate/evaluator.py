from evaluate.test_questions import TEST_DATA
from utils.logger import logger
from rag.initializer import initialize_pipeline
from evaluate.runner import run_test_questions
from evaluate.ragas_runner import run_ragas_evaluation, attach_ragas_metrics
from utils.experiment import create_experiment_metadata
from evaluate.result_writer import add_metadata, save_results
from evaluate.display import display_ragas_results
from evaluate.summary import build_summary
from evaluate.logging import log_df_info, log_avg_scores

def main():

    """Run the complete RAG evaluation pipeline."""

    logger.info("Starting RAG evaluation...")

    # Initialize
    components = initialize_pipeline()
    
    # Run RAG pipeline
    evaluation_results = run_test_questions(
        TEST_DATA, 
        components,
    )

    # Evaluate with RAGAS
    results, ragas_time = run_ragas_evaluation(
        evaluation_results, 
        components,
    )

    # Prepare results
    df = attach_ragas_metrics(
        results, 
        evaluation_results,
    )

    metadata = create_experiment_metadata()
    df = add_metadata(
        df,
        metadata,
    )

    # Display
    display_ragas_results(df)

    summary = build_summary(
        TEST_DATA, 
        evaluation_results, 
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
        evaluation_results,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Evaluation pipeline failed.")
        raise

