import asyncio
import os

import openai

from tinyllm.functions.agent.agent import Agent
from tinyllm.functions.llm.llm_store import LLMStore
from tinyllm.functions.agent.tool_store import ToolStore

openai.api_key = os.environ['OPENAI_API_KEY']


def tt_function(asked_property):
    if asked_property == "name":
        return "Elias"
    elif asked_property == "birthday":
        return "January 1st"


async def run_env():

    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "This is the tool you use retrieve ANY information about the user. Use it to answer questions about his birthday and any personal info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "asked_property": {
                            "type": "string",
                            "enum": ["birthday", "name"],
                            "description": "The specific property the user asked about",
                        },
                    },
                    "required": ["asked_property"],
                }
            }
        },
    ]
    tools_callables = {'test_function': tt_function}
    tool_store = ToolStore(tools=tools,
                           tools_callables=tools_callables)
    llm_store = LLMStore(tool_store=tool_store)
    manager = llm_store.get_llm_function(llm='lite_llm_stream',
                                         name='TinyLLM Agent')
    tiny_env = Agent(llm_store=llm_store,
                     tool_store=tool_store,
                     manager_function=manager)

    async for message in tiny_env.run(content="What is the user's birthday?"):
        print(message)

    return message['message']['completion']

result = asyncio.run(run_env())
print(result)