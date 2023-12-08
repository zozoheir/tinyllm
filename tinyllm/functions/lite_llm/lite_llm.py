import datetime as dt
import json
import os
from datetime import datetime
from typing import Optional, Any
import copy

import openai
from langfuse.model import Usage, CreateGeneration, UpdateGeneration
from litellm import OpenAIError, acompletion

from tinyllm.function import Function
from tinyllm.functions.lite_llm.lite_llm_memory import Memory
from tinyllm.functions.util.example_selector import ExampleSelector
from tinyllm.functions.util.helpers import *
from tinyllm.validator import Validator
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Define which exceptions to retry on
def retry_on_openai_exceptions(exception):
    return isinstance(exception,
                      (openai.error.RateLimitError,
                       openai.error.Timeout,
                       openai.error.ServiceUnavailableError,
                       openai.error.APIError,
                       openai.error.APIConnectionError))


class LiteLLMChatInitValidator(Validator):
    system_prompt: str
    with_memory: bool
    answer_format_prompt: Optional[str]
    example_selector: Optional[ExampleSelector]


class LiteLLMChatInputValidator(Validator):
    message: dict
    model: Optional[str] = 'gpt-3.5-turbo'
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 400
    n: Optional[int] = 1
    stream: Optional[bool] = True


class LiteLLMChatOutputValidator(Validator):
    response: Any


class LiteLLM(Function):
    def __init__(self,
                 system_prompt="You are a helpful assistant",
                 with_memory=True,
                 answer_format_prompt=None,
                 example_selector=None,
                 **kwargs):
        val = LiteLLMChatInitValidator(system_prompt=system_prompt,
                                       with_memory=with_memory,
                                       answer_format_prompt=answer_format_prompt,
                                       example_selector=example_selector,
                                       )
        super().__init__(input_validator=LiteLLMChatInputValidator,
                         **kwargs)
        self.system_prompt = system_prompt
        self.n = 1
        self.memory = Memory(name=f"{self.name}_memory",
                             is_traced=self.is_traced,
                             debug=self.debug,
                             trace=self.trace)
        self.with_memory = with_memory
        self.answer_format_prompt = answer_format_prompt
        self.example_selector = example_selector

        self.total_cost = 0
        # The context builder needs the available token size from the prompt template
        self.answer_format_prompt_size = count_tokens(answer_format_prompt) if answer_format_prompt is not None else 0
        self.completion_args = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    async def run(self, **kwargs):
        message = kwargs['message']
        messages = await self.prepare_messages(
            message=message
        )

        with_tools = 'tool_choice' in kwargs and 'tools' in kwargs
        tools_args = {}
        if with_tools: tools_args = {'tools': kwargs['content']['tools'],
                                     'tool_choice': kwargs['content']['tool_choice']}

        kwargs['messages'] = messages
        api_result = await self.get_completion(
            messages=messages,
            model=kwargs['model'],
            temperature=kwargs['temperature'],
            max_tokens=kwargs['max_tokens'],
            n=kwargs['n'],
            **tools_args
        )

        # Memorize the interaction
        await self.memorize(message=message)
        await self.memorize(message=api_result['choices'][0]['message'])

        return {
            "response": api_result,
        }

    async def memorize(self,
                       message):
        if self.with_memory:
            await self.memory(message=message)


    async def get_completion(self,
                             model,
                             temperature,
                             n,
                             max_tokens,
                             messages,
                             **kwargs):
        self.generation = self.trace.generation(CreateGeneration(
            name=self.name,
            startTime=dt.datetime.now(),
            prompt=messages,
        ))
        api_result = await acompletion(
            model=model,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
        response_message = api_result.model_dump()['choices'][0]['message']
        self.generation.update(UpdateGeneration(
            endTime=dt.datetime.now(),
            completion=response_message,
            usage=Usage(promptTokens=count_tokens(messages), completionTokens=count_tokens(response_message)),
        ))
        return api_result.model_dump()

    async def prepare_messages(self,
                               message):
        system_prompt = get_system_message(content=self.system_prompt)
        examples = []  # add example selector
        if self.example_selector and message['role'] == 'user':
            best_examples = await self.example_selector(input=message['content'])
            for good_example in best_examples['output']['best_examples']:
                examples.append(get_user_message(good_example['USER']))
                examples.append(get_assistant_message(str(good_example['ASSISTANT'])))

        messages = [system_prompt] \
                   + self.memory.get_memories() + \
                   examples + \
                   [message]
        return messages

    @property
    def available_token_size(self):
        memories_size = count_tokens(self.memory.memories,
                                     header='',
                                     ignore_keys=[])

        return (OPENAI_MODELS_CONTEXT_SIZES[
                    self.model] - self.prompt_template.size - memories_size - self.max_tokens - self.answer_format_prompt_size - 10) * 0.99