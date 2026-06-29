import os
from config import DB_PATH, OPENAI_API_KEY, GROQ_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBEDDING_MODEL, SEARCH_TYPE, LLM_MODEL
from datetime import datetime
from datasets import Dataset
import pandas as pd
 
from rag.vectorstore import load_vectorstore
from rag.retriever import create_retriever
from llm.llm import get_llm
from prompts.prompt_template import prompt
from evaluate.test_questions import TEST_DATA
from ragas import evaluate 
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
 
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
 
#ragas_llm = LangchainLLMWrapper(app_llm)
# 1. Initialize the standard OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
 

JUDGE_MODEL = "gpt-4o-mini"
judge_llm = ChatOpenAI(
    model=JUDGE_MODEL,
    api_key=OPENAI_API_KEY,
)

ragas_llm = LangchainLLMWrapper(judge_llm)
# # 2. Use llm_factory to create a compatible InstructorLLM instance
# ragas_llm = llm_factory("gpt-4o-mini", client=openai_client)

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    hf_embeddings
)
# -------------------------------------------------
# 2. Run RAG pipeline on test questions
# -------------------------------------------------
app_llm = ChatOpenAI(
    model= "gpt-5-nano",
    api_key= OPENAI_API_KEY,
    #temperature=0.1
    )

questions = []
answers = []
contexts = []
ground_truths = []

print("Running RAG on test questions...\n")

for item in TEST_DATA:
    question = item["question"]

    docs = retriever.invoke(question.strip())

    retrieved_contexts = [
        doc.page_content for doc in docs
    ]

    context_text = "\n\n".join(retrieved_contexts)

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

    questions.append(question)
    #answers.append(answer)
    answers.append(item["ground_truth"])
    contexts.append(retrieved_contexts)
    ground_truths.append(item["ground_truth"])

    print(f"✅ Q: {question}")
    print(f"   A: {answer[:100]}...\n")

# -------------------------------------------------
# 3. Build RAGAS dataset
# -------------------------------------------------

ragas_dataset = Dataset.from_dict(
    {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }
)
# -------------------------------------------------
# 4. Configure metrics
# -------------------------------------------------
"""
metrics = [
    Faithfulness(
        llm=ragas_llm
    ),
    ResponseGroundedness(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    ),
    ContextPrecisionWithReference(
        llm=ragas_llm
    ),
    ContextRecall(
        llm=ragas_llm
    ),
]
"""
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

results = evaluate(
    dataset=ragas_dataset,
    metrics=metrics,
)
# -------------------------------------------------
# 6. Display results
# -------------------------------------------------

print("\n========== RAGAS SCORES ==========")
print(results)

df = results.to_pandas()

EXPERIMENT_NAME = (
    f"{EMBEDDING_MODEL.split('/')[-1]}"
    f"_chunk{CHUNK_SIZE}"
    f"_overlap{CHUNK_OVERLAP}"
    f"_k{TOP_K}"
    f"_{SEARCH_TYPE}"
)

df["experiment"] = EXPERIMENT_NAME

df["run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df["chunk_size"] = CHUNK_SIZE
df["chunk_overlap"] = CHUNK_OVERLAP
df["top_k"] = TOP_K
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
        "response_relevancy",
        "llm_context_precision_with_reference",
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

df.to_csv(output_path, index=False)

print(f"\n✅ Results saved to {output_path}")

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

#--------Summary CSV--------------------------
summary_row = {
    "experiment": EXPERIMENT_NAME,
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "embedding_model": EMBEDDING_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "top_k": TOP_K,
    "search_type": SEARCH_TYPE,
    "generation_llm": LLM_MODEL,
    "judge_llm": JUDGE_MODEL,
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

for i, doc in enumerate(docs, start=1):
    print(f"\n===== Rank {i} =====")
    print(doc.page_content[:300])   # first 300 characters