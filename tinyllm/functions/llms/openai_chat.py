from typing import Union, List, Any, Dict

import openai

from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator


class OpenAIChatInitValidator(Validator):
    llm_name: str
    temperature: float
    n: int


class LLMCallInputValidator(Validator):
    messages: List[Dict[str, str]]


class OpenAIChat(Function):
    def __init__(self,
                 llm_name,
                 temperature=0.9,
                 n=1,
                 **kwargs):
        val = OpenAIChatInitValidator(llm_name=llm_name,
                                      temperature=temperature,
                                      n=n)
        super().__init__(input_validator=LLMCallInputValidator,
                         **kwargs)
        self.llm_name = llm_name
        self.temperature = temperature
        self.n = n

    async def run_function(self, **kwargs):
        api_result = await openai.ChatCompletion.acreate(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            **kwargs
        )
        return api_result

    async def process_output(self, **kwargs):
        return kwargs['choices'][0]['message']['content']
