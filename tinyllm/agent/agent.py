import json
from typing import Optional

from smartpy.utility.log_util import getLogger
from tinyllm.function import Function
from tinyllm.agent.base import AgentBase

from tinyllm.agent.toolkit import Toolkit
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.llms.llm_store import LLMStore, LLMs
from tinyllm.memory.memory import Memory, BufferMemory
from tinyllm.prompt_manager import PromptManager
from tinyllm.util.helpers import get_openai_message
from tinyllm.tracing.span import langfuse_span
from tinyllm.validator import Validator

logger = getLogger(__name__)


class AgentInitValidator(Validator):
    system_role: str
    llm: Function
    memory: Memory
    toolkit: Optional[Toolkit]
    example_manager: Optional[ExampleManager]


class AgentInputValidator(Validator):
    user_input: str


llm_store = LLMStore()


class Agent(Function):

    def __init__(self,
                 system_role: str,
                 llm: Function = llm_store.get_llm(
                     name='LiteLLM',
                     llm_library=LLMs.LITE_LLM,
                     is_traced=False,
                 ),
                 memory: Memory = BufferMemory(name='Agent memory', is_traced=False),
                 toolkit: Optional[Toolkit] = None,
                 example_manager: Optional[ExampleManager] = ExampleManager(),
                 **kwargs):
        AgentInitValidator(system_role=system_role,
                           llm=llm,
                           toolkit=toolkit,
                           memory=memory,
                           example_manager=example_manager)
        super().__init__(
            input_validator=AgentInputValidator,
            **kwargs)
        self.system_role = system_role
        self.llm = llm
        self.memory = memory
        self.toolkit = toolkit
        self.prompt_manager = PromptManager(
            system_role=system_role,
            example_manager=example_manager,
            memory=memory,
        )

    @langfuse_span(name='User interaction', input_key='user_input',
                   visual_output_lambda=lambda x: x['response']['choices'][0]['message'])
    async def run(self,
                  **kwargs):

        input_msg = get_openai_message(role='user',
                                       content=kwargs['user_input'])

        while True:

            prompt_messages = await self.prompt_manager.format(message=input_msg)
            response_msg = await self.llm(messages=prompt_messages,
                                          tools=self.toolkit.as_dict_list() if self.toolkit else None,
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
