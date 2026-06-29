from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, TEMPERATURE

def get_llm():
        
    llm = ChatOpenAI(
        model= "gpt-5-nano",
        api_key= OPENAI_API_KEY,
        temperature=TEMPERATURE
    )
    return llm