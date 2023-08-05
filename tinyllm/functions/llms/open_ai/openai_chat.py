from datetime import datetime
from typing import Optional
import openai
from langfuse.api.model import CreateTrace, CreateGeneration, Usage, UpdateGeneration
from tinyllm.functions.function import Function
from tinyllm.functions.llms.open_ai.helpers import get_assistant_message, get_user_message, get_openai_api_cost, \
    num_tokens_from_messages
from tinyllm.functions.llms.open_ai.openai_memory import OpenAIMemory
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.validator import Validator


class OpenAIChatInitValidator(Validator):
    llm_name: str
    temperature: float
    prompt_template: OpenAIPromptTemplate  # Prompt template TYPES are validated on a model by model basis
    max_tokens: int
    with_memory: bool


class OpenAIChatInputValidator(Validator):
    message: str


class OpenAIChatOutputValidator(Validator):
    response: str


class OpenAIChat(Function):
    def __init__(self,
                 prompt_template=OpenAIPromptTemplate(name="standard_prompt_template",
                                                      is_traced=False),
                 llm_name='gpt-3.5-turbo',
                 temperature=0,
                 with_memory=False,
                 max_tokens=400,
                 **kwargs):
        val = OpenAIChatInitValidator(llm_name=llm_name,
                                      temperature=temperature,
                                      prompt_template=prompt_template,
                                      max_tokens=max_tokens,
                                      with_memory=with_memory,
                                      )
        super().__init__(input_validator=OpenAIChatInputValidator,
                         **kwargs)
        self.llm_name = llm_name
        self.temperature = temperature
        self.n = 1
        if 'memory' not in kwargs.keys():
            self.memory = OpenAIMemory(name=f"{self.name}_memory",
                                       is_traced=False)
        else:
            self.memory = kwargs['memory']
        self.prompt_template = prompt_template
        self.with_memory = with_memory
        self.max_tokens = max_tokens
        self.total_cost = 0

    @property
    def parameters(self):
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }

    async def add_memory(self, new_memory):
        if self.with_memory is True:
            await self.memory(openai_message=new_memory)

    async def run(self, **kwargs):
        kwargs, call_metadata, messages = await self.prepare_request(
            openai_message=kwargs['message'],
            **kwargs)
        api_result = await self.get_completion(
            model=self.llm_name,
            temperature=self.temperature,
            n=self.n,
            max_tokens=kwargs['max_tokens'],
            messages=messages['messages'],
            call_metadata=call_metadata
        )

        return {'response': api_result}

    async def process_output(self, **kwargs):
        model_response = kwargs['response']['choices'][0]['message']['content']
        await self.add_memory(get_assistant_message(content=model_response))
        return {'response': model_response}



    async def prepare_request(self,
                              openai_message,
                              **kwargs):
        # Format messages into list of dicts for OpenAI
        user_msg = get_user_message(message=openai_message)
        messages = await self.prompt_template(openai_msg=user_msg,
                                              memories=self.memory.memories)
        # add new memory
        await self.add_memory(new_memory=user_msg)

        # Set request args if not provided on __call__
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = self.max_tokens
        # Initiate the call_metadata dictionary
        call_metadata = kwargs.pop('call_metadata', {})
        return kwargs, call_metadata, messages


    async def get_completion(self, **kwargs):
        try:
            # Create tracing generation
            self.llm_trace.create_generation(
                name=f"Assistant response",
                model=self.llm_name,
                prompt=kwargs['messages'],
            )
            # Call OpenAI API
            request_args = {key:value for key,value in kwargs.items() if key not in ['call_metadata']}
            api_result = await openai.ChatCompletion.acreate(**request_args)
            # Update tracing generation
            self.update_openai_generation(api_result=api_result, **kwargs)
            return api_result

        except Exception as e:
            cost = get_openai_api_cost(model=self.llm_name,
                                       completion_tokens=0,
                                       prompt_tokens=num_tokens_from_messages(kwargs['messages']))
            parameters = self.parameters
            parameters['request_cost'] = cost['request_cost']
            parameters['total_cost'] = self.total_cost
            self.llm_trace.generation(UpdateGeneration(
                completion=str({"error": str(e)}),
                metadata={"error": str(e)},
            ))
            raise e


    def update_openai_generation(self,
                                 api_result,
                                 **kwargs):
        # Detach metatadata from kwargs
        call_metadata = kwargs.pop('call_metadata', {})
        # Enrich the result with metadata
        call_metadata['request_params'] = {key: value for key, value in kwargs.items() if key != 'messages'}
        call_metadata['cost_summary'] = get_openai_api_cost(model=self.llm_name,
                                                            completion_tokens=api_result["usage"]['completion_tokens'],
                                                            prompt_tokens=api_result["usage"]['prompt_tokens'])
        # Reattach metatadata for tracing
        # log to Langfuse
        parameters = self.parameters
        if 'max_tokens' in kwargs:
            parameters['max_tokens'] = kwargs['max_tokens']

        call_metadata['cost_summary']['total_cost'] = self.total_cost
        self.llm_trace.update_generation(
            endTime=datetime.now(),
            modelParameters=parameters,
            completion=api_result['choices'][0]['message']['content'],
            metadata=call_metadata,
            usage=Usage(promptTokens=call_metadata['cost_summary']['prompt_tokens'],
                        completionTokens=call_metadata['cost_summary']['completion_tokens'])
        )
        self.total_cost += call_metadata['cost_summary']['request_cost']
