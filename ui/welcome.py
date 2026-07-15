import streamlit as st
from config import (
    WELCOME_TITLE,
    WELCOME_SUBTITLE,
    WELCOME_CAPTION,)

def render_welcome():
    st.header(WELCOME_TITLE)
    st.markdown(WELCOME_SUBTITLE)
    st.caption(WELCOME_CAPTION)
