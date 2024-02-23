import json
from typing import Optional

from smartpy.utility.log_util import getLogger
from tinyllm.agent.agent import AgentInitValidator, AgentInputValidator
from tinyllm.agent.tool import Toolkit

from tinyllm.examples.example_manager import ExampleManager
from tinyllm.function_stream import FunctionStream
from tinyllm.llms.lite_llm_stream import LiteLLMStream
from tinyllm.memory.memory import BufferMemory, Memory
from tinyllm.prompt_manager import PromptManager
from tinyllm.util.helpers import get_openai_message

logger = getLogger(__name__)


class AgentStream(FunctionStream):

    def __init__(self,
                 system_role:  str = 'You are a helpful assistant',
                 example_manager: Optional[ExampleManager] = None,
                 llm: FunctionStream = None,
                 memory: Memory = None,
                 toolkit: Optional[Toolkit] = None,
                 answer_formatting_prompt: Optional[str] = None,
                 tool_retries: int = 3,
                 **kwargs):
        AgentInitValidator(system_role=system_role,
                           llm=llm,
                           toolkit=toolkit,
                           memory=memory,
                           example_manager=example_manager,
                           answer_formatting_prompt=answer_formatting_prompt,
                           tool_retries=tool_retries
                           )
        super().__init__(
            input_validator=AgentInputValidator,
            **kwargs)
        self.system_role = system_role
        self.llm = llm or LiteLLMStream()
        self.toolkit = toolkit
        self.example_manager = example_manager
        self.prompt_manager = PromptManager(
            system_role=system_role,
            example_manager=example_manager,
            memory=memory or BufferMemory(),
            answer_formatting_prompt=answer_formatting_prompt,
        )
        self.tool_retries = tool_retries

    async def run(self,
                  **kwargs):

        input_msg = get_openai_message(
            role='user',
            content=kwargs['content']
        )

        while True:
            kwargs = await self.prompt_manager.prepare_llm_request(message=input_msg,
                                                                   **kwargs)

            async for msg in self.llm(tools=self.toolkit.as_dict_list() if self.toolkit else None,
                                      **kwargs):
                yield msg

            await self.prompt_manager.add_memory(message=input_msg)
            # Process the last message
            if msg['status'] == 'success':
                msg_output = msg['output']

                # Agent decides to call a tool
                if msg_output['type'] == 'tool':
                    tool_call_result_msg = await self.handle_tool_call(msg_output)
                    input_msg = tool_call_result_msg
                elif msg_output['type'] == 'assistant':
                    break

            else:
                raise Exception(msg['message'])

    async def handle_tool_call(self,
                               msg_output):
        api_tool_call = msg_output['last_completion_delta']['tool_calls'][0]
        msg_output['last_completion_delta'].pop('function_call')

        # Memorize tool call with arguments
        json_tool_call = {
            'name': msg_output['completion']['name'],
            'arguments': msg_output['completion']['arguments']
        }
        api_tool_call['function'] = json_tool_call
        msg_output['last_completion_delta']['content'] = ''
        await self.prompt_manager.add_memory(message=msg_output['last_completion_delta'])

        # Memorize tool result
        tool_results = await self.toolkit(
            tool_calls=[{
                'name': api_tool_call['function']['name'],
                'arguments': json.loads(api_tool_call['function']['arguments'])
            }])
        tool_result = tool_results['output']['tool_results'][0]
        tool_call_result_msg = get_openai_message(
            name=tool_result['name'],
            role='tool',
            content=tool_result['content'],
            tool_call_id=api_tool_call['id']
        )
        return tool_call_result_msg
