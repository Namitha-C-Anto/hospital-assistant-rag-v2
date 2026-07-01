import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="RAGAS Evaluation Dashboard",
    layout="wide"
)

st.title("📊 RAGAS Evaluation Dashboard")

csv_path = "ragas_results/all-mpnet-base-v2_chunk800_overlap200_k15_fetch30_lambda0.4_mmr_rerank5_bge-reranker-base_2026-06-30_12_56_55.csv"

try:
    df = pd.read_csv(csv_path)

    st.success("RAGAS results loaded successfully!")

    # ---------------------------------------------------
    # Experiment Summary
    # ---------------------------------------------------

    st.header("Experiment Configuration")

    config_cols = st.columns(4)

    config_cols[0].metric("Embedding", df["embedding_model"].iloc[0])
    config_cols[1].metric("Search", df["search_type"].iloc[0])
    config_cols[2].metric("LLM", df["generation_llm"].iloc[0])
    config_cols[3].metric("Judge", df["judge_llm"].iloc[0])

    config_cols = st.columns(6)

    config_cols[0].metric("Chunk Size", int(df["chunk_size"].iloc[0]))
    config_cols[1].metric("Overlap", int(df["chunk_overlap"].iloc[0]))
    config_cols[2].metric("Top K", int(df["top_k"].iloc[0]))
    config_cols[3].metric("Fetch K", int(df["fetch_k"].iloc[0]))
    config_cols[4].metric("Lambda", df["lambda_mult"].iloc[0])
    config_cols[5].metric(
        "Reranker",
        "Yes" if df["use_reranker"].iloc[0] else "No"
    )

    if df["use_reranker"].iloc[0]:
        st.info(
            f"Reranker Model: {df['reranker_model'].iloc[0]} "
            f"(Top {df['reranker_top_n'].iloc[0]})"
        )

    st.divider()

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    metric_columns = [
        col
        for col in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall"
        ]
        if col in df.columns
    ]

    averages = df[metric_columns].mean().round(3)

    st.header("Average RAGAS Scores")

    cols = st.columns(len(metric_columns))

    for i, metric in enumerate(metric_columns):
        cols[i].metric(metric.replace("_", " ").title(), averages[metric])

    st.divider()

    # ---------------------------------------------------
    # Average Chart
    # ---------------------------------------------------

    avg_df = averages.reset_index()
    avg_df.columns = ["Metric", "Score"]

    fig = px.bar(
        avg_df,
        x="Metric",
        y="Score",
        text="Score",
        range_y=[0, 1],
        title="Average Metric Scores"
    )

    fig.update_traces(texttemplate="%{text:.3f}")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------------------------
    # Per Question
    # ---------------------------------------------------

    st.header("Per Question Analysis")

    selected_metric = st.selectbox(
        "Choose Metric",
        metric_columns
    )

    fig = px.bar(
        df,
        x="user_input",
        y=selected_metric,
        color=selected_metric,
        range_color=[0, 1],
        hover_data=["response"],
        title=f"{selected_metric.replace('_',' ').title()} by Question"
    )

    fig.update_layout(xaxis_tickangle=-40)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------------------------
    # Lowest Scoring Questions
    # ---------------------------------------------------

    st.header("Lowest Scoring Questions")

    metric = st.selectbox(
        "Find lowest scores for",
        metric_columns,
        key="lowest_metric"
    )

    lowest = (
        df[["user_input", metric]]
        .sort_values(metric)
        .head(10)
    )

    st.dataframe(lowest, use_container_width=True)

    st.divider()

    # ---------------------------------------------------
    # Search Question
    # ---------------------------------------------------

    st.header("Search Questions")

    search = st.text_input("Search")

    filtered = df

    if search:
        filtered = df[
            df["user_input"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.dataframe(
        filtered,
        use_container_width=True,
        height=450
    )

except FileNotFoundError:
    st.error("CSV file not found.")