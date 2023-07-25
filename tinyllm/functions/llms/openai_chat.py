import os
from typing import Union, List, Any, Dict

import openai

from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator

class OpenAIChatInitValidator(Validator):
    model_name: str
    temperature: float
    n: int


class LLMCallInputValidator(Validator):
    messages: List[Dict[str, str]]


class OpenAIChat(Function):
    def __init__(self,
                 model_name,
                 temperature=0.9,
                 n=1,
                 **kwargs):
        val = OpenAIChatInitValidator(model_name=model_name,
                                      temperature=temperature,
                                      n=n)
        super().__init__(**kwargs,
                         input_validator=LLMCallInputValidator)
        self.model_name = model_name

    async def run(self, **kwargs):
        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            **kwargs
        )
        return response

    async def process_output(self, **kwargs):
        return kwargs['choices'][0]['message']['content']
