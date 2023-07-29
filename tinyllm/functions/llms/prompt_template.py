import abc
from typing import List, Dict, Any
from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator
from tinyllm.util import prompt_util


class PromptTemplateInitValidator(Validator):
    messages: Any


class PromptTemplate(Function):

    def __init__(self,
                 messages,
                 **kwargs):
        val = PromptTemplateInitValidator(messages=messages)
        super().__init__(**kwargs)
        self.messages = messages

    async def run(self, **kwargs) -> Any:
        formatted_prompt = prompt_util.concatenate_strings(self.messages + [kwargs['message']])
        return formatted_prompt

    @abc.abstractmethod
    async def generate_prompt(self,
                              method='multi',
                              shuffle=False,
                              freeze=[],
                              **kwargs) -> List[str]:
        pass

    @abc.abstractmethod
    async def run(self,
                  **kwargs):
        prompt = await self.generate_prompt(**kwargs)
        return {'prompt': prompt}
