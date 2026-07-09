import os
import pandas as pd
from config import RAGAS_RESULTS_PATH, DATASET, TEMPERATURE, EVALUATION_PATH
from utils.logger import logger
from dataclasses import asdict
import json


def get_results_output_path (metadata):
    output_dir = RAGAS_RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{metadata['experiment']}.json",
    )
    return output_path

def add_metadata(df, metadata):
    for key, value in metadata.items():
        df[key] = value

    return df

#---------------------------------------------------------------------Summary
def save_summary_csv(summary, metadata):
    summary_row = {
        **metadata,
        "dataset": DATASET,
        "temperature": TEMPERATURE,
        **summary,
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

#-------------------------------------------------------------JSON
def save_json_results(
    output_path,
    metadata,
    summary,
    test_data,
    evaluation_results,
):
    json_data = {
        "version": "1.0",
        "experiment": metadata,
        "summary": summary,
        "run_statistics": {
            "total_questions": len(test_data),
            "successful_questions": len(evaluation_results),
            "failed_questions": len(test_data) - len(evaluation_results),
        },
        "results": [
            asdict(result)
            for result in evaluation_results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            json_data,
            f,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(f"✅ JSON saved to {output_path}")

#-------------------------------------------------------------------
def save_results(
    metadata,
    summary,
    TEST_DATA,
    evaluation_results):
    
    save_summary_csv(summary, metadata)
    
    output_path = get_results_output_path(metadata)
    
    save_json_results(output_path, metadata, summary, TEST_DATA, evaluation_results)