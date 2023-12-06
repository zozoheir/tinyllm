import datetime as dt
import json

from smartpy.utility.log_util import getLogger
from tinyllm.function import Function, FunctionStream
from tinyllm.functions.util.helpers import get_openai_message

logger = getLogger(__name__)


class TinyEnvironment(FunctionStream):

    def __init__(self,
                 llm_store,
                 tool_store,
                 manager_llm: str,
                 manager_args: dict,
                 **kwargs):

        super().__init__(**kwargs)

        self.llm_store = llm_store
        self.tool_store = tool_store
        self.tool_store.llm_trace = self.llm_trace
        self.manager = self.llm_store.get_agent(llm=manager_llm,
                                                llm_args=manager_args,
                                                llm_trace=self.llm_trace)

    def initialize_round(self):
        assistant_response = ""
        tool_call = {
            "name": None,
            "arguments": ""
        }
        msg_role = None

        return msg_role, assistant_response, tool_call

    async def run(self,
                  user_input):

        NEXT_AGENT_INPUT = get_openai_message(role='user',
                                              content=user_input)
        while True:

            msg_role, assistant_response, tool_call = self.initialize_round()
            yielded_function_call = False
            async for chunk, completion in self.manager(message=NEXT_AGENT_INPUT,
                                                        tool_choice='auto',
                                                        tools=self.tool_store.tools):
                chunk, assistant_response, tool_call = self.handle_stream_chunks(chunk,
                                                                                 assistant_response,
                                                                                 tool_call)
                if tool_call['name'] is not None and yielded_function_call is False:
                    yield {'type': 'tool',
                           'response': f"Calling function {tool_call['name']} with arguments {tool_call['arguments']}"}
                    yielded_function_call = True
                else:
                    yield {'type': 'user_response', 'response': {"chunk": chunk, "completion": completion}}

            # Case: the Agent decided to call a tool
            if chunk['choices'][0]['finish_reason'] == 'tool_calls':
                tool_call['arguments'] = json.loads(tool_call['arguments'])
                current_tool = tool_call['name']
                tool_msg = await self.llm_store.tool_store.run_tool(current_tool,
                                                                    tool_call['arguments'])
                await self.manager.memory(message=NEXT_AGENT_INPUT)
                NEXT_AGENT_INPUT = tool_msg
            else:

                break

    def handle_stream_chunks(self,
                             chunk,
                             assistant_response,
                             function_call: dict):
        # Function call

        if getattr(chunk['choices'][0]['delta'], 'tool_calls', None):
            assistant_response += chunk.choices[0].delta.tool_calls[0].function.arguments
            function_call['arguments'] += chunk.choices[0].delta.tool_calls[0].function.arguments
            # Get function name
            if getattr(chunk.choices[0].delta, 'tool_calls', None):
                if function_call['name'] is None:
                    function_call['name'] = chunk.choices[0].delta.tool_calls[0].function.name

        # Assistant response
        elif isinstance(getattr(chunk['choices'][0]['delta'], 'content', None), str):
            assistant_response += chunk['choices'][0]['delta'].content

        return chunk, assistant_response, function_call
