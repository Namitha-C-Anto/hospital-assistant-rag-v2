import streamlit as st

def load_css() -> None:
    """Load custom CSS used by the application."""

    st.markdown("""
    <style>

    /* ---------- Sidebar Buttons ---------- */

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;

        text-align: left !important;
        justify-content: flex-start !important;

        padding: 0.15rem 0.35rem;
        min-height: 1rem !important;
        height: 1.5rem !important;

        border-radius: 6px !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button p {
        font-size: 0.89rem !important;
        font-weight: 650 !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] {
        margin-bottom: 0.1rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: rgba(255,255,255,.08) !important;
    }

    /* ---------- Sidebar Text ---------- */

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        font-size: 0.82rem !important;
    }

    /* Expander titles */
    section[data-testid="stSidebar"] summary {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* Captions */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-size: 0.75rem !important;
    }

    </style>
    """, unsafe_allow_html=True)