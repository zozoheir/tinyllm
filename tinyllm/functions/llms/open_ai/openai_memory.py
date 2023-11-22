from typing import Dict

from tinyllm.functions.llms.util.memory import Memory
from tinyllm.functions.llms.open_ai.util.helpers import count_tokens
from tinyllm.functions.validator import Validator


class OpenAIMemoryInputValidator(Validator):
    openai_message: Dict

class OpenAIMemoryOutputValidator(Validator):
    success: bool

class OpenAIMemory(Memory):
    def __init__(self,
                 **kwargs):
        super().__init__(
            input_validator=OpenAIMemoryInputValidator,
            output_validator=OpenAIMemoryOutputValidator,
            **kwargs
        )
        self.memories = []

    async def run(self, **kwargs):
        self.memories.append(kwargs['openai_message'])
        return {'success': True}

    @property
    def size(self):
        return count_tokens(self.memories,
                            header='',
                            ignore_keys=[])
