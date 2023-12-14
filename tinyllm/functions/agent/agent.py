import json
from typing import Union, Optional


from smartpy.utility.log_util import getLogger
from tinyllm.function import Function
from tinyllm.functions.agent.base import AgentBase

from tinyllm.functions.agent.toolkit import Toolkit
from tinyllm.functions.examples.example_manager import ExampleManager
from tinyllm.functions.memory.memory import Memory
from tinyllm.functions.util.helpers import get_openai_message
from tinyllm.util.tracing.span import langfuse_span
from tinyllm.validator import Validator

logger = getLogger(__name__)


class AgentInitValidator(Validator):
    manager_llm: Function
    toolkit: Toolkit
    memory: Union[Memory, None]
    example_manager: Optional[ExampleManager]


class Agent(AgentBase, Function):

    def __init__(self,
                 manager_llm: Function,
                 toolkit: Toolkit,
                 memory=Memory(name='agent_memory', is_traced=False),
                 example_manager=ExampleManager(),
                 **kwargs):
        AgentInitValidator(manager_llm=manager_llm,
                           toolkit=toolkit,
                           memory=memory)
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.manager_llm = manager_llm
        self.memory = memory
        self.example_manager = example_manager

    @langfuse_span(name='User interaction', input_key='user_input')
    async def run(self,
                  user_input,
                  **kwargs):

        input_openai_msg = get_openai_message(role='user',
                                              content=user_input)

        while True:

            messages = await self.prepare_messages(
                message=input_openai_msg,
            )
            msg = await self.manager_llm(messages=messages,
                                         tools=self.toolkit.as_dict_list(),
                                         parent_observation=self.parent_observation)

            if msg['status'] == 'success':
                is_tool_call = msg['output']['response']['choices'][0]['finish_reason'] == 'tool_calls'

                if is_tool_call:
                    # Agent decides to call a tool
                    # TODO When ready, implement parallel function calls

                    # Memorize tool call
                    tool_call_msg = msg['output']['response']['choices'][0]['message']
                    tool_call_msg['content'] = ''
                    tool_calls = tool_call_msg['tool_calls']

                    await self.memorize(message=tool_call_msg,
                                        parent_observation=self.parent_observation)

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
                    input_openai_msg = function_call_msg

                else:
                    # Agent decides to respond
                    return {'response': msg['output']['response']}
            else:
                raise (Exception(msg))


"""

    @property
    def available_token_size(self):
        memories_size = 0
        if self.memory:
            memories_size = count_tokens(self.memory.memories,
                                         header='',
                                         ignore_keys=[])
            system_role_size = count_tokens(get_openai_message(role='system',
                                                               content=self.manager_llm.system_role))
        return (OPENAI_MODELS_CONTEXT_SIZES[
                    self.model] - system_role_size - memories_size - self.max_tokens - self.answer_format_prompt_size - 10) * 0.99

"""
