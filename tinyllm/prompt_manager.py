import os
from enum import Enum
from textwrap import dedent

from tinyllm.examples.example_manager import ExampleManager
from tinyllm.llms.lite_llm import DEFAULT_LLM_MODEL, LLM_TOKEN_LIMITS, DEFAULT_CONTEXT_FALLBACK_DICT
from tinyllm.memory.memory import Memory
from tinyllm.util.helpers import get_openai_message, count_tokens, OPENAI_MODELS_CONTEXT_SIZES
import datetime as dt

from tinyllm.util.message import SystemMessage, UserMessage, Text, AssistantMessage


class MaxTokensStrategy(Enum):
    MAX = 'max_context'
    EXPECTED_RATIO = 'max_user_context_examples'


class PromptManager:
    """
    This class is responsible for formatting the prompt for the LLM and managing:
    - model (to avoid exceeding the token limit)
    - max_tokens (based on the expected completion size)
    """

    def __init__(self,
                 system_role: str,
                 example_manager: ExampleManager,
                 memory: Memory,
                 answer_formatting_prompt: str = None,
                 is_time_aware: bool = True, ):
        self.system_role = system_role
        self.example_manager = example_manager
        self.memory = memory
        self.answer_formatting_prompt = answer_formatting_prompt.strip() if answer_formatting_prompt is not None else None
        self.is_time_aware = is_time_aware

    async def format_messages(self, message):

        current_time = '\n\n\n<Current time: ' + \
                       str(dt.datetime.utcnow()).split('.')[
                           0] + '>'

        system_content = self.system_role + '\n\n' + current_time
        system_msg = SystemMessage(system_content)
        memories = [] if self.memory is None else await self.memory.get_memories()
        examples = []

        if self.example_manager is not None:
            for example in self.example_manager.constant_examples:
                examples.append(example.user_message)
                examples.append(example.assistant_message)

            if self.example_manager.example_selector is not None and message['role'] == 'user':
                best_examples = await self.example_manager.example_selector(input=message['content'])
                for good_example in best_examples['output']['best_examples']:
                    examples.append(UserMessage(good_example['user']))
                    examples.append(AssistantMessage(good_example['assistant']))

        answer_format_msg = [
            UserMessage(self.answer_formatting_prompt)] if self.answer_formatting_prompt is not None else []

        messages = [system_msg] + memories + examples + answer_format_msg + [message]
        return messages

    async def prepare_llm_request(self,
                                  message,
                                  json_model=None,
                                  **kwargs):

        messages = await self.format_messages(message)

        if kwargs['max_tokens_strategy']:
            max_tokens, model = self.get_run_config(messages=messages, **kwargs)
            kwargs['max_tokens'] = max_tokens
            kwargs['model'] = model
        else:
            kwargs['model'] = kwargs.get('model', DEFAULT_LLM_MODEL)
            kwargs['max_tokens'] = kwargs.get('max_tokens', 800)

        kwargs['messages'] = messages
        if json_model:
            kwargs['response_format'] = json_model
        return kwargs

    async def add_memory(self,
                         message):
        if self.memory is not None:
            await self.memory(message=message)

    @property
    async def size(self):
        messages = await self.prepare_llm_request(message=get_openai_message(role='user', content=''))
        return count_tokens(messages)

    def get_run_config(self, messages, **kwargs):

        user_msg = messages[-1]
        user_msg_size = count_tokens(user_msg)
        all_msg_size = count_tokens(messages)
        model = kwargs.get('model', DEFAULT_LLM_MODEL)
        model_token_limit = LLM_TOKEN_LIMITS[model]

        if 'max_tokens' in kwargs:
            max_tokens = kwargs['max_tokens']
        elif kwargs['max_tokens_strategy'] == MaxTokensStrategy.MAX:
            leftover_to_use = model_token_limit - all_msg_size
            max_tokens = min(kwargs.get('allowed_max_tokens', 4096), leftover_to_use)
        elif kwargs['max_tokens_strategy'] == MaxTokensStrategy.EXPECTED_RATIO:
            max_tokens = min(max(800, user_msg_size * kwargs['expected_io_ratio']), 4096)
            expected_total_size = all_msg_size + max_tokens
            if expected_total_size / model_token_limit > 1:
                model = DEFAULT_CONTEXT_FALLBACK_DICT[kwargs['model']]

        return int(max_tokens), model
