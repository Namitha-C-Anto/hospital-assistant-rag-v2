import uuid
from typing import Any

import streamlit as st  

def list_chats():
    """Return all chat sessions."""
    return st.session_state.chat_sessions.items()

#-------------------------------------------------------------

def initialize_chat_sessions() -> None:
    """Initialize chat session state."""

    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = None

#----------------------------------------------------------------
def switch_chat(chat_id: str | None) -> None:
    """Switch to an existing chat or clear the current selection."""
    
    st.session_state.current_chat = chat_id

    # Clear sources when switching chats
    st.session_state.pop("last_pipeline_result", None)
#-------------------------------------------------------------
def create_chat_session() -> str:
    chat_id = str(uuid.uuid4())

    count = len(st.session_state.chat_sessions) + 1

    st.session_state.chat_sessions[chat_id] = {
        "title": "",
        "messages": [],
    }

    st.session_state.current_chat = chat_id

    return chat_id
#-------------------------------------------------------------

def rename_chat(question: str) -> None:
    """Rename a new chat using the first user question."""

    current = st.session_state.current_chat

    if current is None:
        return

    chat = st.session_state.chat_sessions[current]

    if not chat["title"]:
        title = question.strip()

        if len(title) > 30:
            title = title[:27] + "..."

        chat["title"] = title

#-------------------------------------------------------------

def get_chat_history() -> list[dict[str, Any]]:
    """Return the messages of the currently selected chat."""

    current = st.session_state.current_chat

    if current is None:
        return []

    return st.session_state.chat_sessions[current]["messages"]

#-------------------------------------------------------------

def format_chat_history(chat_history: list[dict[str, Any]]) -> str:
    """
    Convert chat history into a string for the prompt.
    """
    return "\n".join(
        f"Human: {chat['question']}\nAI: {chat['answer']}"
        for chat in chat_history
    )
#-------------------------------------------------------------
   
def save_chat(question: str, answer: str) -> None:
    """Save a message to the active chat."""

    current = st.session_state.current_chat

    if current is None:
        create_chat_session()
        current = st.session_state.current_chat

    st.session_state.chat_sessions[current]["messages"].append(
        {
            "question": question,
            "answer": answer,
        }
    )

#-------------------------------------------------------------
def delete_chat(chat_id: str) -> None:
    """Delete a chat session."""

    chats = st.session_state.chat_sessions

    if chat_id in chats:
        del chats[chat_id]

    if st.session_state.current_chat == chat_id:
        if chats:
            # Switch to the first remaining chat
            st.session_state.current_chat = next(iter(chats))
        else:
            # No chats left
            st.session_state.current_chat = None
    
    # Clear sources when switching chats
    st.session_state.pop("last_pipeline_result", None)