from typing import List, Dict, Optional

import openai
from langchain.llms.cohere import completion_with_retry, acompletion_with_retry

from tinyllm.functions.llms.openai.helpers import get_user_message
from tinyllm.functions.llms.llm_call import LLMCall
from tinyllm.functions.llms.openai.openai_memory import OpenAIMemory
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator

from langchain.llms.openai import OpenAIChat as langchain_openai_chat

class OpenAIChatInitValidator(Validator):
    llm_name: str
    temperature: float
    n: int
    prompt_template: Optional[OpenAIPromptTemplate]  # Prompt template TYPES are validated on a model by model basis


class OpenAIChat(LLMCall):
    def __init__(self,
                 prompt_template=OpenAIPromptTemplate(name="standard_prompt_template"),
                 llm_name='gpt-3.5-turbo',
                 temperature=0,
                 n=1,
                 memory=None,
                 functions=[],
                 **kwargs):
        val = OpenAIChatInitValidator(llm_name=llm_name,
                                      temperature=temperature,
                                      n=n,
                                      prompt_template=prompt_template,
                                      memory=memory)
        super().__init__(prompt_template=prompt_template,
                         **kwargs)
        self.llm_name = llm_name
        self.temperature = temperature
        self.n = n
        if memory is None:
            self.memory = OpenAIMemory(name=f"{self.name}_memory")
        else:
            self.memory = memory
        self.prompt_template = prompt_template
        self.functions = functions

    async def generate_prompt(self, message: str):
        return await self.prompt_template(message=message)

    async def run(self, **kwargs):
        message = kwargs.pop('message')
        prompt = await self.generate_prompt(message=message)
        await self.memory(role='user', message=message)

        api_result = acompletion_with_retry(llm=langchain_openai_chat,
                                            model=self.llm_name,
                                            temperature=self.temperature,
                                            n=self.n,
                                            messages=prompt['prompt'],
                                            functions=self.functions,
                                            function_call='auto',
                                            **kwargs
                                            )

        # api_result = await openai.ChatCompletion.acreate()
        return {'response': api_result}

    async def process_output(self, **kwargs):
        model_response = kwargs['response']['choices'][0]['message']['content']
        await self.memory(role='assistant', message=model_response)
        return {'response': model_response}
