import os
from config import OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,FETCH_K, LAMBDA_MULT, USE_RERANKER, RERANKER_MODEL, RERANKER_TOP_N, EMBEDDING_MODEL, SEARCH_TYPE, LLM_MODEL, JUDGE_MODEL, DATASET, TEMPERATURE, RETRIEVAL_MODE
from datetime import datetime
from datasets import Dataset
import pandas as pd
from rag.vectorstore import load_vectorstore
from rag.retriever import create_retriever, retrieve_documents
from prompts.prompt_template import prompt
from evaluate.test_questions import TEST_DATA
from ragas import evaluate 
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from rag.reranker import reranker
from rag.rrf import reciprocal_rank_fusion
import json 
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,  # Newer equivalent of ContextPrecisionWithReference
    ContextRecall
)

# -------------------------------------------------
# 1. Load RAG components
# -------------------------------------------------

vectorstore = load_vectorstore()
retriever = create_retriever(vectorstore)
faiss_retriever = retriever["faiss"]
bm25_retriever = retriever["bm25"]  

#JUDGE_MODEL = LLM_MODEL
judge_llm = ChatOpenAI(
    model=JUDGE_MODEL,
    api_key=OPENAI_API_KEY,
)

ragas_llm = LangchainLLMWrapper(judge_llm) 

hf_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    hf_embeddings
)
# -------------------------------------------------
# 2. Run RAG pipeline on test questions
# -------------------------------------------------
app_llm = ChatOpenAI(
    model= LLM_MODEL,
    api_key= OPENAI_API_KEY,
    )

evaluation_results = []

print("Running RAG on test questions...\n")

DEBUG = False

for item in TEST_DATA:
    question = item["question"]
    try:  
 
        if RETRIEVAL_MODE == "faiss":
            docs = retrieve_documents(question, retriever)
        else:
            docs = retrieve_documents(
                question,
                retriever["faiss"],
                retriever["bm25"],
            )

        ##--------------------DEDUPLICATION
        import hashlib

        seen = set()
        unique_docs = []

        for doc in docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
 
            key = (
                doc.metadata["source"],
                doc.metadata["page"],
                doc.metadata.get("chunk_id", hash(doc.page_content))
            )

            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        docs = unique_docs

        #------------------RERANK
        # Rerank them
        if USE_RERANKER:
            reranked_results = reranker.compress_documents(
                documents=docs,
                query=question
            )
        else:
            reranked_results = docs[:TOP_K]

        retrieved_contexts = [
            doc.page_content for doc in reranked_results
        ]
        
        debug_contexts  = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in reranked_results
        ]

        context_text = "\n\n---\n\n".join(retrieved_contexts)

        messages = prompt.format_messages(
            context=context_text,
            question=question,
            chat_history=""
        )

        response = app_llm.invoke(messages)

        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        evaluation_results.append(
                {
                    "question": question,

                    "retrieval": {
                        "faiss": [
                            {
                                "rank": i + 1,
                                "source": doc.metadata.get("source"),
                                "page": doc.metadata.get("page"),
                                "content": doc.page_content,
                            }
                            for i, doc in enumerate(faiss_docs)
                        ],

                        "bm25": [
                            {
                                "rank": i + 1,
                                "source": doc.metadata.get("source"),
                                "page": doc.metadata.get("page"),
                                "content": doc.page_content,
                            }
                            for i, doc in enumerate(bm25_docs)
                        ],

                        "rrf": [
                            {
                                "rank": i + 1,
                                "source": doc.metadata.get("source"),
                                "page": doc.metadata.get("page"),
                                "content": doc.page_content,
                            }
                            for i, doc in enumerate(docs)
                        ],

                        "reranked": [
                            {
                                "rank": i + 1,
                                "source": doc.metadata.get("source"),
                                "page": doc.metadata.get("page"),
                                "content": doc.page_content,
                            }
                            for i, doc in enumerate(reranked_results)
                        ],
                    },

                    "answer": answer,
                    "reference": item["ground_truth"],
                }
            )

        if DEBUG:
            print("="*80)
            print("QUESTION")
            print(question)

            print("\nRESPONSE")
            print(answer)

            print("\nREFERENCE")
            print(item["ground_truth"])

            print("\nCONTEXT")
            print(context_text)
            print("="*80)

            for i, ctx in enumerate(debug_contexts, 1):
                print(f"\nChunk {i}")
                print(ctx["metadata"])
                print(ctx["content"])
            
    except Exception as e:
        print(f"Failed question: {question}")
        print(e)
            
# -------------------------------------------------
# 3. Build RAGAS dataset
# -------------------------------------------------

ragas_dataset = Dataset.from_dict(
    {
        "user_input": [
            x["question"]
            for x in evaluation_results
        ],

        "response": [
            x["answer"]
            for x in evaluation_results
        ],

        "retrieved_contexts": [
            [
                chunk["content"]
                for chunk in x["retrieval"]["reranked"]
            ]
            for x in evaluation_results
        ],

        "reference": [
            x["reference"]
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

print("\nEvaluating with RAGAS...\n")

if not evaluation_results:
    raise RuntimeError("No successful test cases were collected.")

try:
    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
    )
except Exception as e:
    print("RAGAS evaluation failed")
    print(e)
    raise
# -------------------------------------------------
# 6. Display results
# -------------------------------------------------

print("\n========== RAGAS SCORES ==========")
print(results)

df = results.to_pandas()
for result, (_, row) in zip(evaluation_results, df.iterrows()):

    result["metrics"] = {
        "faithfulness": row["faithfulness"],
        "answer_relevancy": row["answer_relevancy"],
        "context_precision": row["context_precision"],
        "context_recall": row["context_recall"],
    }

EXPERIMENT_NAME = (
    f"{EMBEDDING_MODEL.split('/')[-1]}"
    f"_chunk{CHUNK_SIZE}"
    f"_overlap{CHUNK_OVERLAP}"
    f"_k{TOP_K}"
    f"_fetch{FETCH_K}"
    f"_lambda{LAMBDA_MULT}"
    f"_{SEARCH_TYPE}"
)

if USE_RERANKER:
    EXPERIMENT_NAME += (
        f"_rerank{RERANKER_TOP_N}"
        f"_{RERANKER_MODEL.split('/')[-1]}"
    )
else:
    EXPERIMENT_NAME += "_noreranker"

experiment_info = {
    "experiment": EXPERIMENT_NAME,
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

    "dataset": DATASET,

    "embedding_model": EMBEDDING_MODEL,

    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,

    "search_type": SEARCH_TYPE,
    "top_k": TOP_K,
    "fetch_k": FETCH_K,
    "lambda_mult": LAMBDA_MULT,

    "use_reranker": USE_RERANKER,
    "reranker_model": RERANKER_MODEL,
    "reranker_top_n": RERANKER_TOP_N,

    "generation_llm": LLM_MODEL,
    "judge_llm": JUDGE_MODEL,
}


df["experiment"] = EXPERIMENT_NAME
df["run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df["chunk_size"] = CHUNK_SIZE
df["chunk_overlap"] = CHUNK_OVERLAP
df["top_k"] = TOP_K
df["fetch_k"] = FETCH_K
df["lambda_mult"] = LAMBDA_MULT
df["use_reranker"] = USE_RERANKER
df["reranker_model"] = RERANKER_MODEL
df["reranker_top_n"] = RERANKER_TOP_N
df["embedding_model"] = EMBEDDING_MODEL
df["search_type"] = SEARCH_TYPE
df["generation_llm"] = LLM_MODEL
df["judge_llm"] = JUDGE_MODEL

print("\nAvailable columns:")
print(df.columns.tolist())

print("\n--- Per Question Breakdown ---")

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

print(df[columns_to_show].to_string(index=False))
# -------------------------------------------------
# 7. Save results
# -------------------------------------------------

output_dir = os.path.join("evaluate", "ragas_results")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, f"{EXPERIMENT_NAME}_{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}.csv")

json_path = output_path.replace(".csv", ".json")

json_data = {
    "experiment": experiment_info,
    "results": evaluation_results,
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        json_data,
        f,
        indent=4,
        ensure_ascii=False,
    )

print(f"✅ JSON saved to {json_path}")

# -------------------------------------------------Display
summary = {}

for col in [
    "faithfulness",
    "answer_relevancy",
    "response_relevancy",
    "context_precision",
    "context_recall"
]:
    if col in df.columns:
        summary[col] = round(df[col].mean(), 3)

print(df[[
    "user_input",
    "response",
    "reference",
    "retrieved_contexts",
    "faithfulness"
]])

summary_row = {
    # -----------------------------
    # Experiment Information
    # -----------------------------
    "experiment": EXPERIMENT_NAME,
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

    # -----------------------------
    # Dataset
    # -----------------------------
    "dataset": DATASET,

    # -----------------------------
    # Embedding
    # -----------------------------
    "embedding_model": EMBEDDING_MODEL,

    # -----------------------------
    # Chunking
    # -----------------------------
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,

    # -----------------------------
    # Retrieval
    # -----------------------------
    "search_type": SEARCH_TYPE,
    "top_k": TOP_K,
    "fetch_k": FETCH_K,
    "lambda_mult": LAMBDA_MULT,

    # -----------------------------
    # Reranker
    # -----------------------------
    "use_reranker": USE_RERANKER,
    "reranker_model": RERANKER_MODEL if USE_RERANKER else "None",
    "reranker_top_n": RERANKER_TOP_N if USE_RERANKER else "None",

    # -----------------------------
    # LLM
    # -----------------------------
    "generation_llm": LLM_MODEL,
    "judge_llm": JUDGE_MODEL,
    "temperature": TEMPERATURE,

    # -----------------------------
    # Average RAGAS Scores
    # -----------------------------
    "faithfulness": round(df["faithfulness"].mean(), 3),
    "answer_relevancy": round(df["answer_relevancy"].mean(), 3),
    "context_precision": round(df["context_precision"].mean(), 3),
    "context_recall": round(df["context_recall"].mean(), 3),
}

EVALUATION_PATH = os.path.join("evaluate")
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

print("\n========== AVERAGE SCORES ==========")
for metric, score in summary.items():
    print(f"{metric}: {score}")
