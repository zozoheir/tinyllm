import asyncio
import os
import openai

from tinyllm.app.helpers import tinyllm_agent_functions, tinyllm_agent_prompt_template, function_callables
from tinyllm.functions.llms.open_ai.openai_chat_agent import OpenAIChatAgent

openai.api_key = os.getenv("OPENAI_API_KEY")

loop = asyncio.get_event_loop()

openai_agent = OpenAIChatAgent(
    name="TinyLLM Agent",
    llm_name="gpt-3.5-turbo",
    openai_functions=tinyllm_agent_functions,
    function_callables=function_callables,
    temperature=0,
    prompt_template=tinyllm_agent_prompt_template,
    max_tokens=1000,
    verbose=True,
)


if __name__ == "__main__":
    result = asyncio.run(openai_agent(message="What is tinyllm?"))
    see = result["response"]
    print(see)