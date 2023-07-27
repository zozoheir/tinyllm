from typing import Union, List, Any, Dict

import openai

from tinyllm.functions.function import Function
from tinyllm.functions.llms.openai.helpers import get_user_message
from tinyllm.functions.llms.openai.openai_memory import OpenAIMemory
from tinyllm.functions.validator import Validator


class OpenAIChatInitValidator(Validator):
    llm_name: str
    temperature: float
    n: int
    prompt_template: List[Dict[str, str]]


class LLMCallInputValidator(Validator):
    message: str


class OpenAIChat(Function):
    def __init__(self,
                 llm_name='gpt-3.5-turbo',
                 temperature=0,
                 n=1,
                 prompt_template=[],
                 memory=None,
                 **kwargs):
        val = OpenAIChatInitValidator(llm_name=llm_name,
                                      temperature=temperature,
                                      n=n,
                                      prompt_template=prompt_template,
                                      memory=memory)
        super().__init__(input_validator=LLMCallInputValidator,
                         **kwargs)
        self.llm_name = llm_name
        self.temperature = temperature
        self.n = n
        if memory is None:
            self.memory = OpenAIMemory(name=f"{self.name}_memory")
        else:
            self.memory = memory
        self.prompt_template = prompt_template

    def get_messages(self,
                     user_message):
        open_ai_user_msg = get_user_message(user_message)
        messages = self.prompt_template + self.memory.memories + [open_ai_user_msg]
        return messages


    async def run(self, **kwargs):
        msg = kwargs.pop('message')
        messages = self.get_messages(msg)
        await self.memory(role='user', message=msg)
        api_result = await openai.ChatCompletion.acreate(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            messages=messages,
            **kwargs
        )
        return {'response':api_result}

    async def process_output(self, **kwargs):
        model_response = kwargs['response']['choices'][0]['message']['content']
        await self.memory(role='assistant', message=model_response)
        return {'response': model_response}
