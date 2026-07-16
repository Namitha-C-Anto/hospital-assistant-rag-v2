from collections import OrderedDict
import os
import streamlit as st

def render_sources(last_pipeline_result):

    documents = last_pipeline_result.retrieval_result.retrieved_documents

    grouped = OrderedDict()

    for rank, document in enumerate(documents, start=1):

        source = document.metadata.get("source") or "Unknown PDF"
        pdf_name = os.path.basename(source)

        page = document.metadata.get("page") or "Unknown"

        if pdf_name not in grouped:
            grouped[pdf_name] = []

        grouped[pdf_name].append({
            "rank": rank,
            "page": page,
            "content": document.content
        })

    with st.expander("📚 Retrieved Sources", expanded=False):

        for pdf_name, chunks in grouped.items():

            display_name = (
                os.path.splitext(pdf_name)[0]
                .replace("_", " ")
                .title()
            )

            with st.expander(
                f"📄 {display_name} ({len(chunks)} chunks)",
                expanded=False,
            ):

                for chunk in chunks:

                    st.caption(
                        f"Rank {chunk['rank']} • Page {chunk['page']}"
                    )

                    st.caption(chunk["content"][:200])

                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)