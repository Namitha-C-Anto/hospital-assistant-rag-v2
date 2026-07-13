from typing import Any

import streamlit as st 


def get_chat_history() -> list[dict[str, Any]]:
    """
    Return the chat history stored in the current Streamlit session.
    """
    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    return st.session_state.chat_history

def format_chat_history(chat_history: list[dict[str, Any]]) -> str:
    """
    Convert chat history into a string for the prompt.
    """
    return "\n".join(
        f"Human: {chat['question']}\nAI: {chat['answer']}"
        for chat in chat_history
    )
    
def save_chat(
    question: str, 
    answer: str
) -> None: 

    """
    Save a question-answer pair to the current Streamlit session.
    """

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )