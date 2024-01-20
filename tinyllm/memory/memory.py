from abc import abstractmethod

from tinyllm.function import Function
from tinyllm.util.helpers import count_tokens
from tinyllm.validator import Validator


class MemoryOutputValidator(Validator):
    memories: list

class MemoryInputValidator(Validator):
    message: dict

class Memory(Function):
    def __init__(self,
                 **kwargs):
        super().__init__(
            output_validator=MemoryOutputValidator,
            **kwargs
        )
        self.memories = None

    async def run(self, **kwargs):
        self.memories.append(kwargs['message'])
        return {'memories': self.memories}

    @property
    def size(self):
        return count_tokens(self.memories)

    @abstractmethod
    async def load_memories(self):
        pass

    @abstractmethod
    async def get_memories(self):
        pass


class BufferMemoryInitValidator(Validator):
    buffer_size: int

class BufferMemory(Memory):

    def __init__(self,
                 buffer_size=10,
                 **kwargs):
        BufferMemoryInitValidator(buffer_size=buffer_size)
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.memories = []

    async def get_memories(self):
        return self.memories[-self.buffer_size:]