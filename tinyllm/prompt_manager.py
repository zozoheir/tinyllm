from tinyllm.examples.example_manager import ExampleManager
from tinyllm.memory.memory import Memory
from tinyllm.util.helpers import get_openai_message, count_tokens, OPENAI_MODELS_CONTEXT_SIZES


class PromptManager:

    def __init__(self,
                 system_role: str,
                 example_manager: ExampleManager,
                 memory: Memory,
                 answer_formatting_prompt: str = None):
        self.system_role = system_role
        self.example_manager = example_manager
        self.memory = memory
        self.answer_formatting_prompt = answer_formatting_prompt

    async def format(self,
                     message):

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

        # -- Messages order --
        # system prompt
        # memories
        # constant examples
        # selected (variable) examples
        # Answer formatting prompt
        # input message
        messages = [system_role] + memories + examples + answer_format_msg + [message]

        return messages

    async def add_memory(self,
                         message):
        if self.memory is not None:
            await self.memory(message=message)
