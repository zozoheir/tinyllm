from tinyllm.functions.llms.lite_llm.util.helpers import get_openai_message, count_tokens
from tinyllm.functions.llms.util.memory import Memory
from tinyllm.functions.validator import Validator


class LiteLLMMemoryInputValidator(Validator):
    role: str
    content: str


class LiteLLMMemoryOutputValidator(Validator):
    memories: list


class LiteLLMMemory(Memory):
    def __init__(self,
                 **kwargs):
        super().__init__(
            input_validator=LiteLLMMemoryInputValidator,
            output_validator=LiteLLMMemoryOutputValidator,
            **kwargs
        )
        self.memories = []

    async def run(self, **kwargs):
        msg = get_openai_message(**kwargs)
        self.memories.append(msg)
        return {'memories': self.memories[:-1]}

    @property
    def size(self):
        return count_tokens(self.memories,
                            header='',
                            ignore_keys=[])

    def get_memories(self,
                     method='list'):
        if method == 'list':
            return self.memories
