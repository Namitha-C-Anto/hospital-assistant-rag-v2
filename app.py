import os
import streamlit as st
from config import DB_PATH, APP_TITLE, SUB_TITLE
from rag.vectorstore import load_vectorstore
from rag.retriever import create_retriever
from llm.llm import get_llm
from prompts.prompt_template import prompt
from memory.session_memory import get_chat_history, save_chat
from rag.reranker import reranker

if not os.path.exists(DB_PATH):
    import build_db
    
st.title(APP_TITLE)
st.caption(SUB_TITLE)

vectorstore = load_vectorstore()
retriever = create_retriever(vectorstore)

llm = get_llm()

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

question = st.chat_input(
    "Ask your hospital-related question"
)

if question:

    history_text = "\n".join(
        f"Human: {chat['question']}\n AI: {chat['answer']}"

        for chat in chat_history
    )

    results = retriever.invoke(
        question.strip().upper()
    )

    # Rerank them
    reranked_results = reranker.compress_documents(
        documents=results,
        query=question
    )

    context = "\n\n".join(
        
        doc.page_content 

        for doc in reranked_results
    )

    messages = prompt.format_messages(
        context = context,
        question = question,
        chat_history = history_text
    )

    response = llm.invoke(
        messages
        )

    with st.chat_message("user"):

        st.write(question)

    with st.chat_message("assistant"):

        st.write(response.content)
  
    save_chat(question, response.content)

    with st.expander("Retrieved Context"):
        
        for doc in results:
            
            pdf_name = os.path.basename(
                doc.metadata.get("source", "Unknown PDF")
            ) 
            
            page_no = doc.metadata.get("page", "Unknown Page")
            
            st.write(
                f"📄 PDF: {pdf_name}"
            )
            st.write(
                f"📍Page: {page_no}"
            )

            st.write(doc.page_content[:500])
            st.divider()