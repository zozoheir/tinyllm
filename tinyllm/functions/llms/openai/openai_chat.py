from typing import List, Dict, Optional
import openai

from tinyllm.functions.function import Function
from tinyllm.functions.llms.openai.helpers import get_assistant_message, get_user_message
from tinyllm.functions.llms.openai.openai_memory import OpenAIMemory
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def chat_completion_with_backoff(**kwargs):
    return await openai.ChatCompletion.acreate(**kwargs)


class OpenAIChatInitValidator(Validator):
    llm_name: str = 'gpt-3.5-turbo'
    temperature: float = 0
    prompt_template: Optional[OpenAIPromptTemplate] = OpenAIPromptTemplate(
        name="standard_prompt_template")  # Prompt template TYPES are validated on a model by model basis
    max_tokens: int = 1000

class OpenAIChatInputValidator(Validator):
    message: str

class OpenAIChatOutputValidator(Validator):
    response: str


class OpenAIChat(Function):
    def __init__(self,
                 prompt_template,
                 llm_name,
                 temperature,
                 max_tokens,
                 **kwargs):
        val = OpenAIChatInitValidator(llm_name=llm_name,
                                      temperature=temperature,
                                      prompt_template=prompt_template,
                                      max_tokens=max_tokens)
        super().__init__(input_validator=OpenAIChatInputValidator,
                         **kwargs)
        self.llm_name = llm_name
        self.temperature = temperature
        self.n = 1
        if 'memory' not in kwargs.keys():
            self.memory = OpenAIMemory(name=f"{self.name}_memory")
        else:
            self.memory = kwargs['memory']
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens

    async def add_memory(self, new_memory):
        await self.memory(openai_message=new_memory)

    async def run(self, **kwargs):
        user_msg = get_user_message(message=kwargs['message'])
        messages = await self.prompt_template(openai_msg=user_msg,
                                              memories=self.memory.memories)
        await self.add_memory(new_memory=user_msg)
        api_result = await chat_completion_with_backoff(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            max_tokens=self.max_tokens,
            messages=messages['messages'],
        )
        return {'response': api_result}

    async def process_output(self, **kwargs):
        model_response = kwargs['response']['choices'][0]['message']['content']
        await self.memory(openai_message=get_assistant_message(content=model_response))
        return {'response': model_response}
