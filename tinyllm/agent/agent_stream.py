import json

from smartpy.utility.log_util import getLogger
from tinyllm.agent.agent import AgentInitValidator, AgentInputValidator
from tinyllm.function_stream import FunctionStream
from tinyllm.agent.base import AgentBase

from tinyllm.agent.toolkit import Toolkit
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.memory.memory import BufferMemory
from tinyllm.util.helpers import get_openai_message
from tinyllm.tracing.span import langfuse_span_generator

logger = getLogger(__name__)


class AgentStream(AgentBase, FunctionStream):

    def __init__(self,
                 **kwargs):
        AgentInitValidator(llm=kwargs['llm'],
                           toolkit=kwargs['toolkit'],
                           memory=kwargs['memory'],
                           system_role=kwargs['system_role'],
                           example_manager=kwargs['example_manager'])
        function_attributes = [key for key in kwargs.keys() if key in list(FunctionStream(name='util').__dict__.keys())]
        FunctionStream.__init__(
            self,
            input_validator=AgentInputValidator,
            **{key: kwargs[key] for key in function_attributes})
        AgentBase.__init__(self, **kwargs)

    @langfuse_span_generator(name='User interaction', input_key='user_input',
                             visual_output_lambda=lambda x: x['output']['completion'])
    async def run(self,
                  user_input):

        input_msg = get_openai_message(role='user',
                                       content=user_input)

        while True:
            prompt_messages = await self.prompt_manager.format(message=input_msg)
            await self.prompt_manager.memory(message=input_msg)

            async for msg in self.llm(messages=prompt_messages,
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
                    await self.prompt_manager.memory(message=msg_output['delta'])

                    # Memorize tool result
                    tool_results = await self.toolkit(
                        tool_calls=[{
                            'name': api_tool_call['function']['name'],
                            'arguments': json.loads(api_tool_call['function']['arguments'])
                        }],
                        trace=self.trace)
                    tool_result = tool_results['output']['tool_results'][0]
                    tool_call_result_msg = get_openai_message(
                        name=tool_result['name'],
                        role='tool',
                        content=tool_result['content'],
                        tool_call_id=api_tool_call['id']
                    )
                    input_msg = tool_call_result_msg

                elif msg_output['type'] == 'completion':
                    break
            else:
                raise (Exception(msg))
