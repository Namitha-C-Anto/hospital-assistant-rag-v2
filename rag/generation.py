import time
from typing import Any
from prompts.prompt_template import prompt

def generate_answer(
    question: str,
    context_text: str,
    app_llm: Any,
    chat_history: str | None = None,
) -> tuple[str, dict[str, int], float, float]:
    """
    Generate an answer using the application LLM.

    Args:
        question: User question.
        context_text: Retrieved context passed to the prompt.
        app_llm: Initialized application LLM.
        chat_history: Previous conversation history.

    Returns:
        A tuple containing:
            - Generated answer
            - Token usage statistics
            - Prompt formatting time (seconds)
            - LLM generation time (seconds)
    """

    # -------------------------------------------------
    # Build the prompt with retrieved context and
    # conversation history.
    # -------------------------------------------------
    prompt_start = time.perf_counter()

    messages = prompt.format_messages(
        context=context_text,
        question=question,
        chat_history=chat_history or "",
    )
    prompt_time = round(time.perf_counter() - prompt_start, 4)

    # -------------------------------------------------
    # Generate the response from the LLM.
    # -------------------------------------------------
    generation_start = time.perf_counter()
    response = app_llm.invoke(messages)
    generation_time = round(time.perf_counter() - generation_start,4)
    
    # Extract the generated answer.
    answer = (
        response.content
        if hasattr(response, "content")
        else str(response)
    )

    # Extract token usage information if available.
    usage = getattr(
        response, 
        "response_metadata",
         {},
    ).get("token_usage", {})

    return (
        answer, 
        usage, 
        prompt_time, 
        generation_time
    )
