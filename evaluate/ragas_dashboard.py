import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="RAGAS Evaluation Summary",
    layout="wide",
)

st.title("📊 RAG Evaluation Summary")

# --------------------------------------------------------
# Load comparison CSV
# --------------------------------------------------------

csv_path = "evaluate/evaluation_results/experiment_summary.csv"

df = pd.read_csv(csv_path)

metrics = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

df = df.dropna(subset=metrics)

# --------------------------------------------------------
# Select representative experiments
# --------------------------------------------------------

selected = [
     "faiss_20260706_214041_reranker",
    "hybrid_20260706_215506_no_reranker",
    "hybrid_20260706_215133_reranker",
]

comparison = (
    df[df["experiment"].isin(selected)]
    .copy()
)

comparison["Method"] = comparison["experiment"].map({
    "faiss_20260706_214041_reranker": "FAISS",
    "hybrid_20260706_215506_no_reranker": "Hybrid",
    "hybrid_20260706_215133_reranker": "Hybrid + Reranker",
})

comparison = comparison.sort_values(
    "Method",
    key=lambda s: s.map({
        "FAISS": 0,
        "Hybrid": 1,
        "Hybrid + Reranker": 2,
    }),
)

# --------------------------------------------------------
# Compact Experiment Summary
# --------------------------------------------------------

st.caption(
    f"""
**Dataset:** {comparison['dataset'].iloc[0]} &nbsp;&nbsp;•&nbsp;&nbsp;
**Questions:** {int(comparison['total_questions'].max())} &nbsp;&nbsp;•&nbsp;&nbsp;
**Embedding:** {comparison['embedding_model'].iloc[0].split('/')[-1]} &nbsp;&nbsp;•&nbsp;&nbsp;
**LLM:** {comparison['generation_llm'].iloc[0]} &nbsp;&nbsp;•&nbsp;&nbsp;
**Judge:** {comparison['judge_llm'].iloc[0]}
"""
)

st.divider()

# --------------------------------------------------------
# Evaluation Table
# --------------------------------------------------------

display = comparison[
    [
        "Method",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
].round(3)

display.columns = [
    "Method",
    "Faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
]

st.subheader("RAGAS Evaluation")

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------------
# Comparison Chart
# --------------------------------------------------------

plot_df = display.melt(
    id_vars="Method",
    var_name="Metric",
    value_name="Score",
)

fig = px.bar(
    plot_df,
    x="Metric",
    y="Score",
    color="Method",
    barmode="group",
    text="Score",
    range_y=[0, 1],
)

fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
)

fig.update_layout(
    height=450,
    xaxis_title="",
    yaxis_title="Score",
    legend_title="",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------
# Performance
# --------------------------------------------------------

performance = comparison[
    [
        "Method",
        "avg_pipeline_latency",
        "avg_retrieval_latency",
        "avg_generation_latency",
        "avg_total_tokens",
    ]
].round(2)

performance.columns = [
    "Method",
    "Pipeline (s)",
    "Retrieval (s)",
    "Generation (s)",
    "Tokens",
]

st.subheader("Performance")

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True,
)