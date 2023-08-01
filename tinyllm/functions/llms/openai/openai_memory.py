from typing import Optional, Dict

from tinyllm.functions.llms.memory import Memory
from tinyllm.functions.validator import Validator


class OpenAIMemoryInputValidator(Validator):
    openai_message:Dict

class OpenAIMemory(Memory):
    def __init__(self,
                 **kwargs):
        super().__init__(
            input_validator=OpenAIMemoryInputValidator,
            **kwargs
        )
        self.memories = []

    async def run(self, **kwargs):
        self.memories.append(kwargs['openai_message'])
        return {'success': True}
