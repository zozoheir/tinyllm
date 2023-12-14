import json

from smartpy.utility.log_util import getLogger
from tinyllm.function_stream import FunctionStream
from tinyllm.functions.agent.base import AgentBase

from tinyllm.functions.agent.toolkit import Toolkit
from tinyllm.functions.examples.example_manager import ExampleManager
from tinyllm.functions.memory.memory import Memory
from tinyllm.functions.util.helpers import get_openai_message
from tinyllm.util.tracing.span import langfuse_span, langfuse_span_generator

logger = getLogger(__name__)

class AgentStream(AgentBase, FunctionStream):

    def __init__(self,
                 manager_llm: FunctionStream,
                 toolkit: Toolkit,
                 memory=Memory(name='agent_memory', is_traced=False),
                 example_manager=ExampleManager(),
                 **kwargs):
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.manager_llm = manager_llm
        self.memory = memory
        self.example_manager = example_manager

    @langfuse_span_generator(name='User interaction', input_key='user_input')
    async def run(self,
                  user_input):

        input_openai_msg = get_openai_message(role='user',
                                              content=user_input)
        await self.memorize(message=input_openai_msg)

        while True:
            messages = await self.prepare_messages(
                message=input_openai_msg
            )
            async for msg in self.manager_llm(messages=messages,
                                              tools=self.toolkit.as_dict_list()):
                yield msg

            # Agent decides to call a tool
            if msg['status'] == 'success':
                msg_output = msg['output']
                if msg_output['type'] == 'tool':
                    # TODO When ready, implement parallel function calls
                    api_tool_call = msg_output['delta']['tool_calls'][0]
                    msg_output['delta'].pop('function_call')

                    # Memorize tool call with arguments
                    json_tool_call = {
                        'name': msg_output['completion']['name'],
                        'arguments': msg_output['completion']['arguments']
                    }
                    api_tool_call['function'] = json_tool_call
                    msg_output['delta']['content'] = ''
                    await self.memorize(message=msg_output['delta'])

                    # Memorize tool result
                    tool_results = await self.toolkit(
                        tool_calls=[{
                            'name': api_tool_call['function']['name'],
                            'arguments': json.loads(api_tool_call['function']['arguments'])
                        }],
                        trace=self.trace)
                    tool_result = tool_results['output']['tool_results'][0]
                    function_call_msg = get_openai_message(
                        name=tool_result['name'],
                        role='tool',
                        content=tool_result['content'],
                        tool_call_id=api_tool_call['id']
                    )
                    input_openai_msg = function_call_msg

                elif msg_output['type'] == 'completion':
                    break
            else:
                raise (Exception(msg))
