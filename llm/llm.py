from langchain_openai import ChatOpenAI
from config import (
    OPENAI_API_KEY, 
    TEMPERATURE, 
    LLM_MODEL,)

def get_llm()-> ChatOpenAI:
    """
    Create and return the configured ChatOpenAI model.
    """
        
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE
    )