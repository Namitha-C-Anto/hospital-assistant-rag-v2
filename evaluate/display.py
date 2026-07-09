from utils.logger import logger

def display_ragas_results(df):

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
