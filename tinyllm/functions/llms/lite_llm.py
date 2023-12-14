import datetime as dt
from typing import Optional, Any

from langfuse.model import Usage, CreateGeneration, UpdateGeneration
from litellm import OpenAIError, acompletion

from tinyllm.function import Function
from tinyllm.functions.examples.example_manager import ExampleManager
from tinyllm.functions.memory.memory import Memory
from tinyllm.functions.examples.example_selector import ExampleSelector
from tinyllm.functions.util.helpers import *
from tinyllm.util.tracing.generation import langfuse_generation
from tinyllm.validator import Validator
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type


class LiteLLMChatInitValidator(Validator):
    system_role: str
    answer_format_prompt: Optional[str]


class LiteLLMChatInputValidator(Validator):
    messages: List[Dict]
    model: Optional[str] = 'gpt-3.5-turbo'
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 400
    n: Optional[int] = 1
    stream: Optional[bool] = True


class LiteLLMChatOutputValidator(Validator):
    response: Any


class LiteLLM(Function):
    def __init__(self,
                 system_role="You are a helpful assistant",
                 example_manager=ExampleManager(),
                 memory=None,
                 answer_format_prompt=None,
                 **kwargs):
        LiteLLMChatInitValidator(system_role=system_role,
                                 memory=memory,
                                 answer_format_prompt=answer_format_prompt,
                                 example_manager=example_manager)
        super().__init__(input_validator=LiteLLMChatInputValidator,
                         **kwargs)
        self.system_role = system_role
        self.n = 1
        self.memory = memory
        self.answer_format_prompt = answer_format_prompt
        self.example_manager = example_manager

        self.total_cost = 0
        # The context builder needs the available token size from the prompt template
        self.answer_format_prompt_size = count_tokens(answer_format_prompt) if answer_format_prompt is not None else 0
        self.completion_args = None

    async def run(self, **kwargs):
        with_tools = 'tools' in kwargs
        tools_args = {}
        if with_tools: tools_args = {'tools': kwargs['tools'],
                                     'tool_choice': kwargs.get('tool_choice', 'auto')}
        api_result = await self.get_completion(
            messages=kwargs['messages'],
            model=kwargs['model'],
            temperature=kwargs['temperature'],
            max_tokens=kwargs['max_tokens'],
            n=kwargs['n'],
            parent_observation=kwargs.get('parent_observation', None),
            **tools_args
        )

        # Memorize the interaction
        # await self.memorize(message=message)
        # await self.memorize(message=api_result['choices'][0]['message'])

        return {
            "response": api_result,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    @langfuse_generation()
    async def get_completion(self,
                             model,
                             temperature,
                             n,
                             max_tokens,
                             messages,
                             parent_observation=None,
                             **kwargs):
        api_result = await acompletion(
            model=model,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
        return api_result.model_dump()
