from utils.logger import logger

def log_pipeline_stats(
    question,
    retrieval_time,
    reranker_time,
    generation_time,
    pipeline_time,
):
    logger.info(
        f"{question} | "
        f"Retrieval={retrieval_time}s | "
        f"Reranker={reranker_time}s | "
        f"Generation={generation_time}s | "
        f"Pipeline={pipeline_time}s"
    )

#-------------------------------------------------------------
def log_debug_info(
    question,
    answer,
    reference,
    context_text,
    retrieval_result,
):
    logger.info("=" * 80)

    logger.info("QUESTION")
    logger.info(question)

    logger.info("\nRESPONSE")
    logger.info(answer)

    logger.info("\nREFERENCE")
    logger.info(reference)

    logger.info("\nCONTEXT")
    logger.info(context_text)

    logger.info("=" * 80)

    for i, ctx in enumerate(retrieval_result.reranked_documents, 1):
        logger.info(f"\nChunk {i}")
        logger.info(ctx.metadata)
        logger.info(ctx.content)

#--------------------------------------------------------------------------
def log_df_info(df):
    columns = [
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
        "faithfulness",
    ]

    available_columns = [
        col for col in columns
        if col in df.columns
    ]

    logger.info(df[available_columns].to_string(index=False))

#------------------------------------------------------------------------
def log_avg_scores(summary):
    logger.info("\n========== AVERAGE SCORES ==========")
    
    for metric, score in summary.items():
        logger.info(f"{metric}: {score}")
