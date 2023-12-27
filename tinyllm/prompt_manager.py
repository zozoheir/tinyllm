from tinyllm.examples.example_manager import ExampleManager
from tinyllm.llms.lite_llm import DEFAULT_LLM_MODEL, LLM_TOKEN_LIMITS, DEFAULT_CONTEXT_FALLBACK_DICT
from tinyllm.memory.memory import Memory
from tinyllm.util.helpers import get_openai_message, count_tokens, OPENAI_MODELS_CONTEXT_SIZES
import datetime as dt


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
                 add_current_time: bool = False, ):
        self.system_role = system_role
        self.example_manager = example_manager
        self.memory = memory
        self.answer_formatting_prompt = answer_formatting_prompt
        self.add_current_time = add_current_time

    async def format_messages(self, message):
        system_role = get_openai_message(role='system',
                                         content=self.system_role)
        memories = [] if self.memory is None else await self.memory.get_memories()
        examples = []
        examples += self.example_manager.constant_examples
        if self.example_manager.example_selector.example_dicts and message['role'] == 'user':
            best_examples = await self.example_manager.example_selector(input=message['content'])
            for good_example in best_examples['output']['best_examples']:
                usr_msg = get_openai_message(role='user', content=good_example['user'])
                assistant_msg = get_openai_message(role='assistant', content=good_example['assistant'])
                examples.append(usr_msg)
                examples.append(assistant_msg)

        answer_format_msg = [get_openai_message(role='user',
                                                content=self.answer_formatting_prompt)] if self.answer_formatting_prompt is not None else []
        if self.add_current_time:
            system_role = f'{self.system_role} \nThe current time is:{str(dt.datetime.utcnow())}'
        messages = [system_role] + memories + examples + answer_format_msg + [message]
        return messages


    async def format(self,
                     message,
                     **kwargs):

        messages = await self.format_messages(message)
        max_tokens, model = self.get_run_config(model=kwargs.get('model', DEFAULT_LLM_MODEL),
                                                prompt_to_completion_multiplier=kwargs.get(
                                                    'prompt_to_completion_multiplier', 1),
                                                input_size=count_tokens(messages))

        return {
            'messages': messages,
            'max_tokens': max_tokens,
            'model': model,
        }

    async def add_memory(self,
                         message):
        if self.memory is not None:
            await self.memory(message=message)

    @property

    async def size(self):
        messages = await self.format(message=get_openai_message(role='user', content=''))
        return count_tokens(messages)


    def get_run_config(self, input_size, prompt_to_completion_multiplier, model):
        model_token_limit = LLM_TOKEN_LIMITS[model]
        context_size_available = model_token_limit - input_size

        max_tokens = max(500, input_size * prompt_to_completion_multiplier)
        expected_total_size = input_size + max_tokens
        if expected_total_size > context_size_available:
            model = DEFAULT_CONTEXT_FALLBACK_DICT[model]
        return max_tokens, model
