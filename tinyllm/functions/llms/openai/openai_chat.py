from typing import List, Dict, Optional

import openai

from tinyllm.functions.llms.openai.helpers import get_user_message
from tinyllm.functions.llms.llm_call import LLMCall
from tinyllm.functions.llms.openai.openai_memory import OpenAIMemory
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator

import openai  # for OpenAI API calls
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def chat_completion_with_backoff(**kwargs):
    return await openai.ChatCompletion.acreate(**kwargs)


class OpenAIChatInitValidator(Validator):
    llm_name: str ='gpt-3.5-turbo'
    temperature: float=0
    n: int=1
    prompt_template: Optional[OpenAIPromptTemplate] = OpenAIPromptTemplate(name="standard_prompt_template") # Prompt template TYPES are validated on a model by model basis

class OpenAIChat(LLMCall):
    def __init__(self,
                 prompt_template,
                 llm_name,
                 temperature,
                 n,
                 **kwargs):
        val = OpenAIChatInitValidator(llm_name=llm_name,
                                      temperature=temperature,
                                      n=n,
                                      prompt_template=prompt_template)
        super().__init__(prompt_template=prompt_template,
                         **kwargs)
        self.llm_name = llm_name
        self.temperature = temperature
        self.n = n
        if 'memory' not in kwargs.keys():
            self.memory = OpenAIMemory(name=f"{self.name}_memory")
        else:
            self.memory = kwargs['memory']
        self.prompt_template = prompt_template

    async def generate_prompt(self, message: str):
        return await self.prompt_template(message=message)

    async def run(self, **kwargs):
        message = kwargs.pop('message')
        role = kwargs.pop('role','user')
        prompt = await self.generate_prompt(message=message)
        await self.memory(role=role, message=message)

        api_result = await chat_completion_with_backoff(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            messages=prompt['prompt'],
            **kwargs
        )
        return {'response': api_result}

    async def process_output(self, **kwargs):
        model_response = kwargs['response']['choices'][0]['message']['content']
        await self.memory(role='assistant', message=model_response)
        return {'response': model_response}
