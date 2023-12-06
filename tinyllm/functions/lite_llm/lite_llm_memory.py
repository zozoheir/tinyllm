from typing import Dict

from tinyllm.functions.util.helpers import count_tokens
from tinyllm.functions.util.memory import Memory
from tinyllm.validator import Validator


class LiteLLMMemoryInputValidator(Validator):
    message: Dict


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
        self.memories.append(kwargs['message'])
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
