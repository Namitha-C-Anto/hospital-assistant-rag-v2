import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="RAGAS Evaluation Dashboard",
    layout="wide"
)

st.title("📊 RAGAS Evaluation Dashboard")

csv_path = "ragas_results.csv"

try:
    df = pd.read_csv(csv_path)

    st.success("RAGAS results loaded successfully!")

    st.subheader("Evaluation Results")

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

    # --------------------------
    # Metric columns
    # --------------------------

    metric_columns = [
        col for col in [
            "faithfulness",
            "answer_relevancy",
            "response_relevancy",
            "context_precision",
            "context_recall"
        ]
        if col in df.columns
    ]

    if metric_columns:

        st.subheader("Average Scores")

        avg_scores = (
            df[metric_columns]
            .mean()
            .round(3)
            .reset_index()
        )

        avg_scores.columns = [
            "Metric",
            "Score"
        ]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                avg_scores,
                use_container_width=True
            )

        with col2:

            fig = px.bar(
                avg_scores,
                x="Metric",
                y="Score",
                title="Average Metric Scores"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # --------------------------
        # Per Question Chart
        # --------------------------

        st.subheader("Per Question Metrics")

        selected_metric = st.selectbox(
            "Select Metric",
            metric_columns
        )

        question_chart = px.bar(
            df,
            x="user_input",
            y=selected_metric,
            title=f"{selected_metric} by Question"
        )

        st.plotly_chart(
            question_chart,
            use_container_width=True
        )

except FileNotFoundError:
    st.error(
        "ragas_results.csv not found. Run evaluation first."
    )