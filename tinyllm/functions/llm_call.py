from typing import Union, List, Any, Dict

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
        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            **kwargs
        )
        return response['choices'][0]['content']
