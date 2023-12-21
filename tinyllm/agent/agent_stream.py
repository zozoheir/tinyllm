import json
from typing import Optional

from smartpy.utility.log_util import getLogger
from tinyllm.agent.agent import AgentInitValidator, AgentInputValidator
from tinyllm.function import Function

from tinyllm.agent.toolkit import Toolkit
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.function_stream import FunctionStream
from tinyllm.memory.memory import BufferMemory, Memory
from tinyllm.prompt_manager import PromptManager
from tinyllm.tracing.langfuse_context import observation, streaming_observation
from tinyllm.util.helpers import get_openai_message

logger = getLogger(__name__)


class AgentStream(FunctionStream):

    def __init__(self,
                 system_role: str,
                 llm: FunctionStream,
                 memory: Memory = BufferMemory(name='Agent memory', is_traced=False),
                 toolkit: Optional[Toolkit] = None,
                 example_manager: Optional[ExampleManager] = None,
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
        self.example_manager = example_manager
        self.prompt_manager = PromptManager(
            system_role=system_role,
            example_manager=example_manager,
            memory=memory,
        )

    @streaming_observation(observation_type='span')
    async def run(self,
                  user_input,
                  **kwargs):

        input_msg = get_openai_message(role='user',
                                       content=user_input)

        while True:
            prompt_messages = await self.prompt_manager.format(message=input_msg)
            await self.prompt_manager.memory(message=input_msg)

            async for msg in self.llm(messages=prompt_messages,
                                      tools=self.toolkit.as_dict_list() if self.toolkit else None,
                                      **kwargs):
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
                        trace=self.parent_observation)
                    tool_result = tool_results['output']['tool_results'][0]
                    tool_call_result_msg = get_openai_message(
                        name=tool_result['name'],
                        role='tool',
                        content=tool_result['content'],
                        tool_call_id=api_tool_call['id']
                    )
                    input_msg = tool_call_result_msg

                elif msg_output['type'] == 'assistant':
                    break
            else:
                raise (Exception(msg))
