from typing import Dict

from tinyllm.function import Function
from tinyllm.functions.util.helpers import count_tokens
from tinyllm.functions.util.memory import Memory
from tinyllm.validator import Validator


class MemoryInputValidator(Validator):
    message: Dict


class MemoryOutputValidator(Validator):
    memories: list


class Memory(Function):
    def __init__(self,
                 **kwargs):
        super().__init__(
            input_validator=MemoryInputValidator,
            output_validator=MemoryOutputValidator,
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
