from tinyllm.function import Function
from tinyllm.functions.helpers import count_tokens
from tinyllm.validator import Validator


class MemoryOutputValidator(Validator):
    memories: list


class Memory(Function):
    def __init__(self,
                 **kwargs):
        super().__init__(
            output_validator=MemoryOutputValidator,
            **kwargs
        )
        self.memories = []

    async def run(self, **kwargs):
        self.memories.append(kwargs['message'])
        return {'memories': self.memories}

    @property
    def size(self):
        return count_tokens(self.memories,
                            header='',
                            ignore_keys=[])

    def get_memories(self,
                     method='list'):
        if method == 'list':
            return self.memories
