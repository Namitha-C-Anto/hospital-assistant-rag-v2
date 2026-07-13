import os
import streamlit as st
from config import (
    DB_PATH, 
    APP_TITLE, 
    SUB_TITLE,)
from utils.logger import logger     
from rag.pipeline import run_rag_pipeline  
from rag.initializer import initialize_rag 
from memory.session_memory import (
    get_chat_history, 
    format_chat_history, 
    save_chat,)
from rag.builder import build_vector_database
from rag.models import PipelineComponents

# -------------------------------------------------
# Load and cache initialized RAG components.
# This prevents reloading the vector database and
# models on every Streamlit rerun.
# -------------------------------------------------
@st.cache_resource
def load_rag_components() -> PipelineComponents:
    logger.info("Initializing RAG components...")  
    return initialize_rag()

def main() -> None:
    """
    Run the Streamlit Hospital Policy RAG application.
    """

    # -------------------------------------------------
    # Create the vector database on first launch if it
    # does not already exist.
    # -------------------------------------------------
    if not os.path.exists(DB_PATH):
        build_vector_database()
        
    # -------------------------------------------------
    # Display the application header.
    # -------------------------------------------------
    st.title(APP_TITLE)
    st.caption(SUB_TITLE)

    # -------------------------------------------------
    # Load cached RAG components.
    # -------------------------------------------------
    rag_components = load_rag_components()

    # -------------------------------------------------
    # Restore and display previous chat messages.
    # -------------------------------------------------
    chat_history = get_chat_history()

    for chat in chat_history:

        with st.chat_message("user"):
            st.write(
                chat["question"]
            )
        with st.chat_message("assistant"):
            st.write(
                chat["answer"]
            )

    # -------------------------------------------------
    # Accept a new user question.
    # -------------------------------------------------
    question = st.chat_input(
        "Ask your hospital-related question"
    )

    if question:
        
        logger.info(f"Processing question: {question}")

        # -------------------------------------------------
        # Format conversation history and execute the
        # complete RAG pipeline.
        # -------------------------------------------------
        history_text = format_chat_history(chat_history)

        pipeline_result = run_rag_pipeline(
                question,
                rag_components,
                chat_history = history_text,
            ) 

        # -------------------------------------------------
        # Display the latest conversation.
        # -------------------------------------------------
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(pipeline_result.answer)
    
        # Save conversation for future turns.
        save_chat(question, pipeline_result.answer)

        # -------------------------------------------------
        # Display retrieved document chunks used to
        # generate the answer.
        # -------------------------------------------------
        with st.expander("Retrieved Context"):
            
            for document in pipeline_result.retrieval_result.retrieved_documents:
                
                pdf_name = os.path.basename(
                    document.metadata.get("source", "Unknown PDF")
                ) 
                
                page_no = document.metadata.get("page", "Unknown Page")
                
                st.write(
                    f"📄 PDF: {pdf_name}"
                )
                st.write(
                    f"📍Page: {page_no}"
                )

                st.write(document.content[:500])
                st.divider()

if __name__ == "__main__":
    main()