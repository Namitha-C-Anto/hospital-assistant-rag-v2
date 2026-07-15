import streamlit as st
from config import APP_TITLE
from memory.chat_manager import (
    switch_chat,
    list_chats,
    delete_chat,)

def render_sidebar() -> None:
    st.markdown(APP_TITLE, unsafe_allow_html=True)

    #Step 2: Create a new chat
    if st.button("＋  New chat", use_container_width=True):
        
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
            st.caption(" ")

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