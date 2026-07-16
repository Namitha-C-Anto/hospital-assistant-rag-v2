import streamlit as st
from config import APP_TITLE
from memory.chat_manager import (
    switch_chat,
    list_chats,
    delete_chat,)

def render_sidebar() -> None:
    st.markdown(APP_TITLE, unsafe_allow_html=True)

    # -----------------------------
    # New Chat
    # -----------------------------

    #Step 2: Create a new chat
    if st.button(
        "➕ New Chat",
        use_container_width=True,
        type="primary",
    ):        
        switch_chat(None)
        st.rerun()



    with st.expander("Recent", expanded=True):

        chats = list(list_chats())

        if not chats:
            st.caption("No conversations yet.")

        #Step 3: Show the active chat
        for chat_id, chat in  chats:
                    
            title = chat["title"]

            if chat_id == st.session_state.current_chat:
                title = f"🩺{title}"

            if st.button(
                title,
                key=chat_id,
                use_container_width=True,
                type="secondary",
            ):
                switch_chat(chat_id)
                st.rerun()

        if st.session_state.current_chat is not None:
            st.divider() 

            if st.button(
                "🗑️ Delete Chat",
                use_container_width=True,
            ):
                st.session_state.confirm_delete = True

            if st.session_state.get("confirm_delete", False):
                st.warning("Delete this chat?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Yes"):
                        delete_chat(st.session_state.current_chat)
                        st.session_state.confirm_delete = False
                        st.rerun()

                with col2:
                    if st.button("Cancel"):
                        st.session_state.confirm_delete = False
                        st.rerun()

    # ---------------------------------------------------
    # Settings
    # ---------------------------------------------------

    with st.expander("⚙️ Settings"):

        provider = st.selectbox(
            "LLM Provider",
            ["OpenAI", "Groq"],
            key="llm_provider",
        )
        st.session_state["provider"] = provider.lower()

        if provider == "OpenAI":
            model = st.selectbox(
                "Model",
                [
                    "gpt-4o-mini",
                    "gpt-5.4-mini",
                ],
                key="llm_model",
            )

            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                key="openai_api_key",
                placeholder="sk-...",
            )

        else:
            model = st.selectbox(
                "Model",
                [
                    "llama-3.3-70b-versatile",
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "qwen/qwen3-32b",
                ],
                key="llm_model",
            )

            api_key = st.text_input(
                "Groq API Key",
                type="password",
                key="groq_api_key",
                placeholder="gsk_...",
            )
        st.session_state["model"] = model
        st.session_state["api_key"] = api_key
 
