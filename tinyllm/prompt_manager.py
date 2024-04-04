import os
from textwrap import dedent

from tinyllm.examples.example_manager import ExampleManager
from tinyllm.llms.lite_llm import DEFAULT_LLM_MODEL, LLM_TOKEN_LIMITS, DEFAULT_CONTEXT_FALLBACK_DICT
from tinyllm.memory.memory import Memory
from tinyllm.util.helpers import get_openai_message, count_tokens, OPENAI_MODELS_CONTEXT_SIZES
import datetime as dt

from tinyllm.util.message import SystemMessage, UserMessage, Text, AssistantMessage


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

        complete_message_note = "Notes:\n- if the last message is yours and looks incomplete, finish it"
        current_time = '\n\n\n<Current time: ' + \
                       str(dt.datetime.utcnow()).split('.')[
                           0] + '>'

        system_content = self.system_role + '\n\n' + complete_message_note + current_time
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
                                  **kwargs):

        messages = await self.format_messages(message)
        if 'expected_io_ratio' in kwargs:
            expected_io_ratio = kwargs.pop('expected_io_ratio', 1)
            max_tokens, model = self.get_run_config(model=kwargs.get('model', DEFAULT_LLM_MODEL),
                                                    expected_io_ratio=expected_io_ratio,
                                                    messages=messages)
            kwargs['max_tokens'] = max_tokens
            kwargs['model'] = model
        else:
            kwargs['model'] = kwargs.get('model', DEFAULT_LLM_MODEL)
            kwargs['max_tokens'] = kwargs.get('max_tokens', 600)

        kwargs['messages'] = messages
        return kwargs

    async def add_memory(self,
                         message):
        if self.memory is not None:
            await self.memory(message=message)

    @property
    async def size(self):
        messages = await self.prepare_llm_request(message=get_openai_message(role='user', content=''))
        return count_tokens(messages)

    def get_run_config(self, messages, expected_io_ratio, model):

        user_msg = messages[-1]
        user_msg_size = count_tokens(user_msg)
        all_msg_size = count_tokens(messages)
        model_token_limit = LLM_TOKEN_LIMITS[model]

        max_tokens = min(max(600, user_msg_size * expected_io_ratio), 4096)
        expected_total_size = all_msg_size + max_tokens
        if expected_total_size / model_token_limit > 1:
            model = DEFAULT_CONTEXT_FALLBACK_DICT[model]

        return int(max_tokens), model
