from abc import abstractmethod

import openai

from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator

class LLMCallInitValidator(Validator):
    provider_name: str


class LLMCall(Function):
    def __init__(self,
                 provider_name,
                 **kwargs):
        val = LLMCallInitValidator(provider_name=provider_name)
        super().__init__(**kwargs)
        self.api_response = None
        self.provider_name = provider_name

    async def run(self, **kwargs):
        return await self.llm_call(**kwargs)

    @abstractmethod
    async def llm_call(self, **kwargs):
        self.api_response = await self.provider(**kwargs)
        return self.api_response
