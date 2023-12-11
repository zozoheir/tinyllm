import datetime as dt
from typing import Optional, Any

from langfuse.model import Usage, CreateGeneration, UpdateGeneration
from litellm import OpenAIError, acompletion

from tinyllm.function import Function
from tinyllm.functions.examples.example_manager import ExampleManager
from tinyllm.functions.llm.memory import Memory
from tinyllm.functions.examples.example_selector import ExampleSelector
from tinyllm.functions.helpers import *
from tinyllm.validator import Validator
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type


class LiteLLMChatInitValidator(Validator):
    system_role: str
    example_manager: Optional[ExampleManager]
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
                 system_role="You are a helpful assistant",
                 example_manager=ExampleManager(),
                 with_memory=True,
                 answer_format_prompt=None,
                 **kwargs):
        LiteLLMChatInitValidator(system_role=system_role,
                                 with_memory=with_memory,
                                 answer_format_prompt=answer_format_prompt,
                                 example_manager=example_manager)
        super().__init__(input_validator=LiteLLMChatInputValidator,
                         **kwargs)
        self.system_role = system_role
        self.n = 1
        self.memory = Memory(name=f"{self.name}_memory",
                             is_traced=self.is_traced,
                             debug=self.debug)
        self.with_memory = with_memory
        self.answer_format_prompt = answer_format_prompt
        self.example_manager = example_manager

        self.total_cost = 0
        # The context builder needs the available token size from the prompt template
        self.answer_format_prompt_size = count_tokens(answer_format_prompt) if answer_format_prompt is not None else 0
        self.completion_args = None

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


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    async def get_completion(self,
                             model,
                             temperature,
                             n,
                             max_tokens,
                             messages,
                             **kwargs):
        self.generation = self.trace.generation(CreateGeneration(
            name=self.name,
            startTime=dt.datetime.utcnow(),
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
            endTime=dt.datetime.utcnow(),
            completion=response_message,
            usage=Usage(promptTokens=count_tokens(messages), completionTokens=count_tokens(response_message)),
        ))
        return api_result.model_dump()

    async def prepare_messages(self,
                               message):
        # system prompt
        # memories
        # constant examples
        # selected examples
        # input message
        system_role = get_system_message(content=self.system_role)
        examples = self.example_manager.constant_examples
        if self.example_manager.example_selector.example_dicts and message['role'] == 'user':
            best_examples = await self.example_manager.example_selector(input=message['content'])
            for good_example in best_examples['output']['best_examples']:
                examples.append(get_user_message(good_example['USER']))
                examples.append(get_assistant_message(str(good_example['ASSISTANT'])))



        messages = [system_role] \
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
