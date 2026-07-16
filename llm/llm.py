from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from config import TEMPERATURE, LLM_MODEL, OPENAI_API_KEY, GROQ_API_KEY


def get_llm(
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
):
    provider = provider.lower()

    if provider == "groq":
        return ChatGroq(
            model=model,
            api_key=api_key or GROQ_API_KEY,
            temperature=TEMPERATURE,
        )

    return ChatOpenAI(
        model=model or LLM_MODEL,
        api_key=api_key or OPENAI_API_KEY,
        temperature=TEMPERATURE,
    )