import os
from typing import Union, List, Any, Dict

import openai

from tinyllm.functions.function import Function
from tinyllm.functions.llm_call import LLMCall
from tinyllm.functions.validator import Validator


class OpenAIChatInitValidator(Validator):
    llm_name: str
    temperature: float
    n: int


class LLMCallInputValidator(Validator):
    messages: List[Dict[str, str]]


class OpenAIChat(LLMCall):
    def __init__(self,
                 llm_name,
                 temperature=0.9,
                 n=1,
                 **kwargs):
        val = OpenAIChatInitValidator(llm_name=llm_name,
                                      temperature=temperature,
                                      n=n)
        super().__init__(provider_name='openai',
                         input_validator=LLMCallInputValidator,
                         **kwargs)
        self.llm_name = llm_name
        self.temperature = temperature
        self.n = n

    async def llm_call(self, **kwargs):
        return await openai.ChatCompletion.acreate(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            **kwargs
        )

    async def process_output(self, **kwargs):
        return kwargs['choices'][0]['message']['content']
