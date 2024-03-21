from abc import abstractmethod
from typing import Union

from tinyllm.function import Function
from tinyllm.util.helpers import count_tokens
from tinyllm.util.message import Message
from tinyllm.validator import Validator


class MemoryOutputValidator(Validator):
    memories: list


class MemoryInputValidator(Validator):
    message: Message


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

        memories_to_return = []
        msg_count = 0
        # Make sure we keep complete tool calls msgs
        for memory in self.memories[::-1]:
            memories_to_return.append(memory)
            if 'tool_calls' in memory.dict() or memory.role == 'tool':
                continue
            else:
                msg_count += 1

            if msg_count == self.buffer_size:
                break

        return memories_to_return[::-1]
