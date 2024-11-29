import json
import pprint
from abc import abstractmethod
from typing import Optional, Union, List, Type, Callable

from pydantic import BaseModel

from tinyllm.agent.tool import Toolkit
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.function import Function
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.memory.memory import Memory, BufferMemory
from tinyllm.prompt_manager import PromptManager, MaxTokensStrategy
from tinyllm.util.message import Content, UserMessage, ToolMessage, AssistantMessage
from tinyllm.validator import Validator


class AgentInitValidator(Validator):
    system_role: str
    llm: Optional[Function]
    memory: Optional[Memory]
    toolkit: Optional[Toolkit]
    example_manager: Optional[ExampleManager]
    initial_user_message_text: Optional[str]
    tool_retries: Optional[int]
    output_model: Optional[Type[BaseModel]]
    prompt_manager: Optional[PromptManager]


class AgentInputValidator(Validator):
    content: Union[str, list, Content, List[Content]]
    max_tokens_strategy: Optional[MaxTokensStrategy] = None
    allowed_max_tokens: Optional[int] = int(4096 * 0.25)
    expects_block: Optional[str] = None


class Brain(BaseModel):
    personality: Optional[str]

    @abstractmethod
    def update(self, **kwargs):
        pass


class AgentCallBackHandler:

    async def on_tools(self,
                       **kwargs):
        pass


class Agent(Function):

    def __init__(self,
                 system_role: str = 'You are a helpful assistant',
                 example_manager: Optional[ExampleManager] = None,
                 llm: Function = None,
                 memory: Memory = None,
                 toolkit: Optional[Toolkit] = None,
                 initial_user_message_text: Optional[str] = None,
                 prompt_manager: Optional[PromptManager] = None,
                 tool_retries: int = 3,
                 output_model: Optional[Type[BaseModel]] = None,
                 brain: Brain = None,
                 **kwargs):

        AgentInitValidator(system_role=system_role,
                           llm=llm,
                           toolkit=toolkit,
                           memory=memory,
                           example_manager=example_manager,
                           initial_user_message_text=initial_user_message_text,
                           tool_retries=tool_retries,
                           output_model=output_model,
                           brain=brain,
                           prompt_manager=None)
        super().__init__(
            input_validator=AgentInputValidator,
            **kwargs
        )
        self.system_role = system_role.strip()
        self.output_model = output_model
        self.llm = llm or LiteLLM()
        self.toolkit = toolkit
        self.example_manager = example_manager
        self.prompt_manager = PromptManager(
            system_role=self.system_role,
            example_manager=example_manager,
            memory=memory or BufferMemory() if toolkit else None,
            initial_user_message_text=initial_user_message_text,
        ) if  prompt_manager is None else prompt_manager
        self.tool_retries = tool_retries
        self.is_stuck = False
        self.brain = brain
        self.session_tool_messages = []

    @property
    def tools(self):
        if self.is_stuck:
            return None
        else:
            return self.toolkit.as_dict_list() if self.toolkit else None

    async def run(self,
                  **kwargs):

        input_msgs = [UserMessage(kwargs['content'])]

        while True:  # Loop until agent decides to respond

            request_kwargs = await self.prompt_manager.prepare_llm_request(messages=input_msgs,
                                                                           json_model=self.output_model,
                                                                           **kwargs)
            response_msg = await self.llm(tools=self.tools,
                                          **request_kwargs)

            for msg in input_msgs:
                await self.prompt_manager.add_memory(message=msg)

            if response_msg['status'] == 'success':
                is_tool_call = response_msg['output']['response']['choices'][0]['message'] == 'tool_calls' or \
                               response_msg['output']['response']['choices'][0]['message'].get('tool_calls',
                                                                                               None) is not None

                if is_tool_call:
                    # Agent decides to call a tool

                    # Memorize tool call
                    tool_call_msg = response_msg['output']['response']['choices'][0]['message']
                    await self.prompt_manager.add_memory(message=AssistantMessage(content='',
                                                                                  tool_calls=tool_call_msg[
                                                                                      'tool_calls']))

                    tool_calls = tool_call_msg.get('tool_calls', [])
                    tool_results = await self.toolkit(
                        tool_calls=[{
                            'name': tool_call['function']['name'],
                            'arguments': json.loads(tool_call['function']['arguments'])
                        } for tool_call in tool_calls]
                    )


                    # Format for next openai call
                    tool_call_messages = [ToolMessage(name=tool_result['name'],
                                                      content=pprint.pformat(tool_result['content']),
                                                      tool_call_id=tool_call['id']) for tool_result, tool_call in
                                          zip(tool_results['output']['tool_results'], tool_calls)]

                    # Set next input
                    input_msgs = tool_call_messages

                else:

                    # Agent decides to respond
                    if self.output_model is None:
                        msg_content = response_msg['output']['response']['choices'][0]['message']['content']
                        await self.prompt_manager.add_memory(
                            message=AssistantMessage(msg_content)
                        )
                        return {'response': response_msg['output']['response']}
                    else:
                        msg_content = response_msg['output']['response']['choices'][0]['message']['content']
                        msg_content = msg_content.replace('```json', '').replace('```', '')
                        parsed_output = json.loads(msg_content)
                        await self.prompt_manager.add_memory(
                            message=AssistantMessage(msg_content)
                        )
                        return {'response': self.output_model(**parsed_output)}

            else:
                raise (Exception(response_msg))

    def is_tool_stuck(self, session_tool_results):
        if len(session_tool_results) < self.tool_retries:
            return False
        retry_window_tool_results = session_tool_results[len(session_tool_results) - self.tool_retries:]
        all_results_the_same = all(
            [tool_results == retry_window_tool_results[0] for tool_results in retry_window_tool_results])
        return all_results_the_same
