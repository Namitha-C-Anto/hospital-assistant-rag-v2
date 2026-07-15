import os
import streamlit as st
from config import (
    DB_PATH, 
    APP_TITLE, 
    WELCOME_TITLE,
    WELCOME_SUBTITLE,
    WELCOME_CAPTION,)
from ui import example_questions
from utils.logger import logger     
from rag.pipeline import run_rag_pipeline  
from rag.initializer import initialize_rag 

from memory.chat_manager import (
    initialize_chat_sessions,
    rename_chat,
    get_chat_history, 
    format_chat_history, 
    save_chat,
    switch_chat,
    list_chats,)
from rag.builder import build_vector_database
from rag.models import PipelineComponents
from ui.styles import load_css
from ui.sidebar import render_sidebar
from ui.welcome import render_welcome
from ui.example_questions import render_example_questions
from ui.sources import render_sources

load_css()
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
    
    #Step 1: Initialize the session state
    initialize_chat_sessions()
        
    # -------------------------------------------------
    # Create the vector database on first launch if it
    # does not already exist.
    # -------------------------------------------------
    if not os.path.exists(DB_PATH):
        build_vector_database()
        
    # -------------------------------------------------
    # Display the Side bar
    # -------------------------------------------------
    with st.sidebar:
        render_sidebar()
        
        if "last_pipeline_result" in st.session_state:
            render_sources(
                st.session_state.last_pipeline_result
            )
    # -------------------------------------------------
    # Load cached RAG components.
    # -------------------------------------------------
    rag_components = load_rag_components()

    # -------------------------------------------------
    # Restore and display previous chat messages.
    # -------------------------------------------------
    chat_history = get_chat_history()

    if not chat_history:
        render_welcome()

    for chat in chat_history:

        with st.chat_message("user"):
            st.write(
                chat["question"]
            )
        with st.chat_message("assistant"):
            st.write(
                chat["answer"]
            )

    #------------------------------------------------------------    
    if not chat_history:
        st.markdown("##### 💡 Try asking")
        
        example_questions = render_example_questions()
        columns = st.columns(2)

        for i, q in enumerate(example_questions):
            with columns[i % 2]:
                if st.button(q, use_container_width=True):
                    st.session_state.selected_question = q

    # -------------------------------------------------
    # Accept a new user question.
    # -------------------------------------------------
    question = st.chat_input(
        "Ask your hospital-related question"
    )

    # If an example question was clicked, use it instead
    if "selected_question" in st.session_state:
        question = st.session_state.pop("selected_question")

    if question:
         
        logger.info(f"Processing question: {question}")

        # -------------------------------------------------
        # Format conversation history and execute the
        # complete RAG pipeline.
        # -------------------------------------------------
        history_text = format_chat_history(chat_history)

        # -------------------------------------------------
        # Display the latest conversation.
        # -------------------------------------------------
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                pipeline_result = run_rag_pipeline(
                    question,
                    rag_components,
                    chat_history=history_text,
                )
            st.write(pipeline_result.answer)
            st.session_state.last_pipeline_result = pipeline_result
    
        # Save conversation for future turns.
        save_chat(question, pipeline_result.answer)
        rename_chat(question)
                  
        st.rerun()
            
if __name__ == "__main__":
    main()