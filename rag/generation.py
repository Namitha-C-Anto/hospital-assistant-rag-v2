import time
from prompts.prompt_template import prompt

def generate_answer(
    question,
    context_text,
    app_llm,
):
    
        prompt_start = time.perf_counter()

        messages = prompt.format_messages(
            context=context_text,
            question=question,
            chat_history=""
        )
        prompt_time = round(time.perf_counter() - prompt_start, 4)

        generation_start = time.perf_counter()
        response = app_llm.invoke(messages)
        generation_time = round(time.perf_counter() - generation_start,4)

        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        usage = getattr(response, "response_metadata", {}).get("token_usage", {})

        return answer, usage, prompt_time, generation_time
