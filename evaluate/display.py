import pandas as pd

from utils.logger import logger

def display_ragas_summary(
    df: pd.DataFrame
)-> None:
    """
    Display a per-question summary of the RAGAS evaluation results.

    Logs the available DataFrame columns and prints the selected
    evaluation metrics for each question. Only columns present in the
    DataFrame are displayed.

    Args:
        df: DataFrame containing the RAGAS evaluation results.
    """

    # -------------------------------------------------
    # Display the available DataFrame columns
    # -------------------------------------------------
    logger.info("\nAvailable columns:")
    logger.info(df.columns.tolist())
    
    # -------------------------------------------------
    # Display the per-question evaluation summary
    # -------------------------------------------------
    logger.info("\n--- Per Question Breakdown ---")

    # -------------------------------------------------
    # Select the evaluation columns to display
    # -------------------------------------------------
    display_columns = [
        col 
        for col in [
            "user_input",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        if col in df.columns
    ]

    # -------------------------------------------------
    # Log the per-question evaluation summary
    # -------------------------------------------------
    logger.info(df[display_columns].to_string(index=False))
