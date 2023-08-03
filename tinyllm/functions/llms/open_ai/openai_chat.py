from datetime import datetime
from typing import List, Dict, Optional
import openai
from langfuse.api.model import CreateTrace, CreateGeneration, Usage

from tinyllm.functions.function import Function
from tinyllm.functions.llms.open_ai.helpers import get_assistant_message, get_user_message, get_openai_api_cost,\
    num_tokens_from_messages
from tinyllm.functions.llms.open_ai.openai_memory import OpenAIMemory
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator

from tinyllm.langfuse import langfuse_client
import logging

logger = logging.getLogger()


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
                 trace=None,
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
        self.total_cost = 0
        if trace is None and self.verbose is True:
            self.trace = langfuse_client.trace(CreateTrace(
                name=self.name,
                userId="test",
                metadata={
                    "model": self.llm_name,
                    "modelParameters": self.parameters,
                    "prompt": self.prompt_template.messages,
                }
            ))
        else:
            self.trace = trace

    @property
    def parameters(self):
        return {
            "temperature": self.temperature,
            "maxTokens": self.max_tokens,
            "n": self.n,
        }

    async def get_completion(self, **kwargs):
        start_time = datetime.now()
        try:
            logger.info(f"openai.ChatCompletion is : {openai.ChatCompletion}")
            api_result = await openai.ChatCompletion.acreate(**kwargs)
            api_result['cost_summary'] = get_openai_api_cost(model=self.llm_name,
                                                             completion_tokens=api_result["usage"]['completion_tokens'],
                                                             prompt_tokens=api_result["usage"]['prompt_tokens'])
            # log to Langfuse
            if self.verbose is True:
                parameters = self.parameters
                parameters['request_cost'] = api_result['cost_summary']['request_cost']
                parameters['total_cost'] = self.total_cost
                self.trace.generation(CreateGeneration(
                    name=f"Assistant response",
                    startTime=start_time,
                    endTime=datetime.now(),
                    model=self.llm_name,
                    modelParameters=parameters,
                    prompt=kwargs['messages'],
                    completion=api_result['choices'][0]['message']['content'],
                    metadata=api_result,
                    usage=Usage(promptTokens=api_result['cost_summary']['prompt_tokens'],
                                completionTokens=api_result['cost_summary']['completion_tokens']),
                ))
            api_result['cost_summary'] = get_openai_api_cost(completion_tokens=api_result["usage"]['completion_tokens'],
                                                             prompt_tokens=api_result["usage"]['prompt_tokens'],
                                                             model=self.llm_name)
            self.total_cost += api_result['cost_summary']['request_cost']
            return api_result

        except Exception as e:
            if self.verbose is True:
                cost = get_openai_api_cost(model=self.llm_name,
                                                             completion_tokens=0,
                                                             prompt_tokens=num_tokens_from_messages(kwargs['messages']))
                parameters = self.parameters
                parameters['request_cost'] = cost['request_cost']
                parameters['total_cost'] = self.total_cost
                self.trace.generation(CreateGeneration(
                    name=f"Assistant",
                    startTime=start_time,
                    endTime=datetime.now(),
                    model=self.llm_name,
                    modelParameters=self.parameters,
                    prompt=kwargs['messages'],
                    completion="ERROR",
                    metadata={"error": str(e)},
                    usage=Usage(promptTokens=cost['prompt_tokens'],
                                completionTokens=cost['completion_tokens']),
                ))
            raise e


    async def add_memory(self, new_memory):
        await self.memory(openai_message=new_memory)

    async def run(self, **kwargs):
        user_msg = get_user_message(message=kwargs['message'])
        messages = await self.prompt_template(openai_msg=user_msg,
                                              memories=self.memory.memories)
        await self.add_memory(new_memory=user_msg)
        api_result= await self.get_completion(
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

