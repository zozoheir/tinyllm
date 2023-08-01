import asyncio
import os
import openai

from tinyllm.copilot_agent.helpers import openai_functions, tinyllm_agent_prompt_template, function_callables
from tinyllm.functions.llms.openai.openai_chat_agent import OpenAIChatAgent

openai.api_key = os.getenv("OPENAI_API_KEY")

loop = asyncio.get_event_loop()

openai_agent = OpenAIChatAgent(
    name="TinyLLM Agent",
    llm_name="gpt-3.5-turbo",
    openai_functions=openai_functions,
    function_callables=function_callables,
    temperature=0,
    prompt_template=tinyllm_agent_prompt_template,
    n=1,
    verbose=True,
)


if __name__ == "__main__":
    openai_msg = {'user_content': "What is tinyllm?"}
    result = asyncio.run(openai_agent(**openai_msg))
    see = result["response"]
