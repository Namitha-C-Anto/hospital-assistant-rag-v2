from datasets import Dataset
from config import USE_RERANKER
import time
from utils.logger import logger
from ragas import evaluate 
from rag.models import Metrics
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,  # Newer equivalent of ContextPrecisionWithReference
    ContextRecall
)

def _build_ragas_dataset(evaluation_results):
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
    return ragas_dataset

#-------------------------------------------------------------------------------

def run_ragas_evaluation(evaluation_results, components):
      
    metrics = [
        Faithfulness(llm=components.ragas_llm),
        ResponseRelevancy(llm=components.ragas_llm, embeddings=components.ragas_embeddings),
        ContextPrecision(llm=components.ragas_llm),
        ContextRecall(llm=components.ragas_llm)
    ]

    if not evaluation_results:
        raise RuntimeError("No successful test cases were collected.")

    logger.info("Evaluating with RAGAS...")
    
    ragas_dataset = _build_ragas_dataset(evaluation_results)  

    try:
        ragas_start = time.perf_counter()
        
        results = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
        )
        ragas_time = round(time.perf_counter() - ragas_start, 4)

    except Exception:
        logger.exception("RAGAS evaluation failed") 
        raise

    
    logger.info("========== RAGAS SCORES ==========")
    logger.info(results)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(results)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    return results, ragas_time

#----------------------------------------------------------------------
def attach_ragas_metrics(results, evaluation_results):

    df = results.to_pandas()

    for result, (_, row) in zip(evaluation_results, df.iterrows()):
        result.metrics = Metrics(
            faithfulness=row["faithfulness"],
            answer_relevancy=row["answer_relevancy"],
            context_precision=row["context_precision"],
            context_recall=row["context_recall"],
        )
    
    return df