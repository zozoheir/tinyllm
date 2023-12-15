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
        # system prompt
        # memories
        # constant examples
        # selected examples
        # input message
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


        messages = [system_role] \
                   + memories + \
                   examples + \
                   [message]

        if self.answer_formatting_prompt is not None:
            messages += [get_openai_message(role='user', content=self.answer_formatting_prompt)]

        return messages


    @property
    def available_token_size(self):
        memories_size = 0
        if self.memory:
            memories_size = count_tokens(self.memory.memories,
                                         header='',
                                         ignore_keys=[])
            system_role_size = count_tokens(get_openai_message(role='system',
                                                               content=self.llm.system_role))
        return (OPENAI_MODELS_CONTEXT_SIZES[
                    self.model] - system_role_size - memories_size - self.max_tokens - self.answer_format_prompt_size - 10) * 0.99
