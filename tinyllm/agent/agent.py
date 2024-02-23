import json
from typing import Optional, Union

from smartpy.utility.log_util import getLogger
from tinyllm.agent.tool import Toolkit
from tinyllm.function import Function

from tinyllm.examples.example_manager import ExampleManager
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.memory.memory import Memory, BufferMemory
from tinyllm.prompt_manager import PromptManager
from tinyllm.util.helpers import get_openai_message
from tinyllm.validator import Validator


class AgentInitValidator(Validator):
    system_role: str
    llm: Optional[Function]
    memory: Optional[Memory]
    toolkit: Optional[Toolkit]
    example_manager: Optional[ExampleManager]
    answer_formatting_prompt: Optional[str]
    tool_retries: Optional[int]


class AgentInputValidator(Validator):
    content: Union[str, list]



class Agent(Function):

    def __init__(self,
                 system_role: str = 'You are a helpful assistant',
                 example_manager: Optional[ExampleManager] = None,
                 llm: Function = None,
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
                           tool_retries=tool_retries)
        super().__init__(
            input_validator=AgentInputValidator,
            **kwargs)
        self.system_role = system_role.strip()
        self.llm = llm or LiteLLM()
        self.toolkit = toolkit
        self.example_manager = example_manager
        self.prompt_manager = PromptManager(
            system_role=self.system_role,
            example_manager=example_manager,
            memory=memory or BufferMemory(),
            answer_formatting_prompt=answer_formatting_prompt,
        )
        self.tool_retries = tool_retries
        self.is_stuck = False
        self.session_tool_calls = []

    @property
    def tools(self):
        if self.is_stuck:
            return None
        else:
            return self.toolkit.as_dict_list() if self.toolkit else None

    async def run(self,
                  **kwargs):

        input_msg = get_openai_message(
            role='user',
            content=kwargs['content']
        )
        session_tool_results = []

        while True:

            request_kwargs = await self.prompt_manager.prepare_llm_request(message=input_msg,
                                                                           **kwargs)

            response_msg = await self.llm(tools=self.tools,
                                          **request_kwargs)
            await self.prompt_manager.add_memory(message=input_msg)

            if response_msg['status'] == 'success':
                is_tool_call = response_msg['output']['response']['choices'][0]['finish_reason'] == 'tool_calls'

                if is_tool_call:
                    # Agent decides to call a tool
                    # TODO When ready, implement parallel function calls

                    # Memorize tool call
                    tool_call_msg = response_msg['output']['response']['choices'][0]['message']
                    tool_call_msg['content'] = ''
                    tool_calls = tool_call_msg['tool_calls']
                    await self.prompt_manager.add_memory(message=tool_call_msg)

                    # Memorize tool result
                    tool_call = tool_calls[0]
                    tool_results = await self.toolkit(
                        tool_calls=[{
                            'name': tool_call['function']['name'],
                            'arguments': json.loads(tool_call['function']['arguments'])
                        }]
                    )
                    session_tool_results.append(tool_results)
                    self.is_stuck = self.is_tool_stuck(session_tool_results)

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
                    msg_content = response_msg['output']['response']['choices'][0]['message']['content']
                    await self.prompt_manager.add_memory(
                        message=get_openai_message(role='assistant', content=msg_content)
                    )
                    return {'response': response_msg['output']['response']}
            else:
                raise (Exception(response_msg))

    def is_tool_stuck(self, session_tool_results):
        if len(session_tool_results) < self.tool_retries:
            return False
        retry_window_tool_results = session_tool_results[len(session_tool_results) - self.tool_retries:]
        all_results_the_same = all(
            [tool_results == retry_window_tool_results[0] for tool_results in retry_window_tool_results])
        return all_results_the_same
