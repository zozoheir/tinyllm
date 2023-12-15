import json
from typing import Optional

from smartpy.utility.log_util import getLogger
from tinyllm.function import Function
from tinyllm.agent.base import AgentBase

from tinyllm.agent.toolkit import Toolkit
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.memory.memory import Memory
from tinyllm.util.helpers import get_openai_message
from tinyllm.tracing.span import langfuse_span
from tinyllm.validator import Validator

logger = getLogger(__name__)


class AgentInitValidator(Validator):
    system_role: str
    llm: Function
    toolkit: Toolkit
    memory: Memory
    example_manager: Optional[ExampleManager]


class AgentInputValidator(Validator):
    user_input: str


class Agent(AgentBase, Function):

    def __init__(self,
                 **kwargs):
        AgentInitValidator(llm=kwargs['llm'],
                           toolkit=kwargs['toolkit'],
                           memory=kwargs['memory'],
                           system_role=kwargs['system_role'],
                           example_manager=kwargs['example_manager'])
        function_attributes = [key for key in kwargs.keys() if key in list(Function(name='util').__dict__.keys())]
        Function.__init__(
            self,
            input_validator=AgentInputValidator,
            **{key: kwargs[key] for key in function_attributes})
        AgentBase.__init__(self, **kwargs)


    @langfuse_span(name='User interaction', input_key='user_input',
                   visual_output_lambda=lambda x: x['response']['choices'][0]['message'])
    async def run(self,
                  **kwargs):

        input_msg = get_openai_message(role='user',
                                       content=kwargs['user_input'])

        while True:

            prompt_messages = await self.prompt_manager.format(message=input_msg)
            response_msg = await self.llm(messages=prompt_messages,
                                          tools=self.toolkit.as_dict_list(),
                                          parent_observation=self.parent_observation)
            await self.prompt_manager.memory(message=input_msg)

            if response_msg['status'] == 'success':
                is_tool_call = response_msg['output']['response']['choices'][0]['finish_reason'] == 'tool_calls'

                if is_tool_call:
                    # Agent decides to call a tool
                    # TODO When ready, implement parallel function calls

                    # Memorize tool call
                    tool_call_msg = response_msg['output']['response']['choices'][0]['message']
                    tool_call_msg['content'] = ''
                    tool_calls = tool_call_msg['tool_calls']

                    await self.prompt_manager.memory(message=tool_call_msg)

                    # Memorize tool result
                    tool_call = tool_calls[0]
                    tool_results = await self.toolkit(
                        tool_calls=[{
                            'name': tool_call['function']['name'],
                            'arguments': json.loads(tool_call['function']['arguments'])
                        }],
                        parent_observation=self.parent_observation)
                    tool_result = tool_results['output']['tool_results'][0]
                    function_call_msg = get_openai_message(
                        name=tool_result['name'],
                        role='tool',
                        content=tool_result['content'],
                        tool_call_id=tool_call['id']
                    )

                    # Set next input
                    input_msg = function_call_msg

                else:
                    # Agent decides to respond
                    return {'response': response_msg['output']['response']}
            else:
                raise (Exception(response_msg))

