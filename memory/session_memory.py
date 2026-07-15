from typing import Any

import streamlit as st 
from memory.chat_manager import create_chat


def get_chat_history() -> list[dict[str, Any]]:
    """Return the messages of the currently selected chat."""

    current = st.session_state.current_chat

    if current is None:
        return []

    return st.session_state.chat_sessions[current]["messages"]


def format_chat_history(chat_history: list[dict[str, Any]]) -> str:
    """
    Convert chat history into a string for the prompt.
    """
    return "\n".join(
        f"Human: {chat['question']}\nAI: {chat['answer']}"
        for chat in chat_history
    )
   
def save_chat(question: str, answer: str) -> None:
    """Save a message to the active chat."""

    current = st.session_state.current_chat

    if current is None:
        create_chat()
        current = st.session_state.current_chat

    st.session_state.chat_sessions[current]["messages"].append(
        {
            "question": question,
            "answer": answer,
        }
    )