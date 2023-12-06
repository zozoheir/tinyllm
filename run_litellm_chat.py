import asyncio
import os

import openai

from tinyllm.functions.environment.environment import TinyEnvironment
from tinyllm.functions.environment.llm_store import LLMStore
from tinyllm.functions.environment.tool_store import ToolStore
from tinyllm.functions.lite_llm.lite_llm_chat import LiteLLMStream, LiteLLM

openai.api_key = os.environ['OPENAI_API_KEY']


def tt_function(asked_property):
    if asked_property == "name":
        return "Elias"
    elif asked_property == "birthday":
        return "January 1st"


"""

    async def process_output(self, **kwargs):

        # Case if OpenAI decides function call
        if kwargs['response']['choices'][0]['finish_reason'] == 'function_call':
            # Call the function
            self.llm_trace.create_span(
                name=f"Calling function: {kwargs['response']['choices'][0]['message']['function_call']['name']}",
                startTime=datetime.now(),
                metadata={'api_result': kwargs['response']['choices'][0]},
            )
            # Call the function
            function_name = kwargs['response']['choices'][0]['message']['function_call']['name']
            function_result = await self.run_agent_function(
                function_call_message=kwargs['response']['choices'][0]['message']['function_call']
            )
            # Append function result to memory
            function_msg = get_function_message(
                content=function_result,
                name=function_name
            )
            # Generate input messages with the function call content
            messages = await self.generate_messages_history(role='function',
                                                            name=function_name,
                                                            content=function_msg['content'])

            # Make API call with the function call content
            # Remove functions arg to get final assistant response
            api_result = await self.get_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.n,
            )
            assistant_response = api_result['choices'][0]['message']['content']
        else:
            # If no function call, just return the result
            assistant_response = kwargs['response']['choices'][0]['message']['content']
            msg = get_openai_message(role='assistant',
                                     content=assistant_response)
            await self.memory(**msg)

        return {'response': assistant_response}

"""


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
    tiny_env = TinyEnvironment(name='TinyLLM Environment',
                               llm_store=llm_store,
                               tool_store=tool_store,
                               manager_llm='lite_llm_stream',
                               manager_args={
                                   'name': 'manager',
                               })

    async for message in tiny_env(user_input="What is the user's birthday?"):
        print(message)


result = asyncio.run(run_env())
print(result)
