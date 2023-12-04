import json
import os
from datetime import datetime
from typing import Optional, Any
import copy

import openai
from langfuse.model import Usage
from litellm import OpenAIError, acompletion

from tinyllm.functions.function import Function
from tinyllm.functions.llms.lite_llm.lite_llm_memory import LiteLLMMemory
from tinyllm.functions.llms.lite_llm.util.helpers import *
from tinyllm.functions.llms.util.example_selector import ExampleSelector
from tinyllm.functions.validator import Validator
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Define which exceptions to retry on
def retry_on_openai_exceptions(exception):
    return isinstance(exception,
                      (openai.error.RateLimitError,
                       openai.error.Timeout,
                       openai.error.ServiceUnavailableError,
                       openai.error.APIError,
                       openai.error.APIConnectionError))


class LiteLLMChatInitValidator(Validator):
    system_prompt: str
    with_memory: bool
    model: str
    temperature: float
    max_tokens: int
    answer_format_prompt: Optional[str]
    openai_functions: Optional[Any] = None
    function_callables: Optional[Any] = None
    example_selector: Optional[ExampleSelector] = None


class LiteLLMChatInputValidator(Validator):
    content: str
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    call_metadata: Optional[dict]
    with_functions: Optional[bool]


class LiteLLMChatOutputValidator(Validator):
    response: Any


class LiteLLMChat(Function):
    def __init__(self,
                 system_prompt="You are a helpful assistant",
                 model='gpt-3.5-turbo',
                 temperature=0,
                 with_memory=False,
                 max_tokens=400,
                 answer_format_prompt=None,
                 openai_functions=None,
                 function_callables=None,
                 example_selector=None,
                 **kwargs):
        val = LiteLLMChatInitValidator(system_prompt=system_prompt,
                                       model=model,
                                       temperature=temperature,
                                       max_tokens=max_tokens,
                                       with_memory=with_memory,
                                       answer_format_prompt=answer_format_prompt,
                                       openai_functions=openai_functions,
                                       function_callables=function_callables,
                                        example_selector=example_selector,
                                       **kwargs
                                       )
        super().__init__(input_validator=LiteLLMChatInputValidator,
                         output_validator=LiteLLMChatOutputValidator,
                         **kwargs)
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.n = 1
        if 'memory' not in kwargs.keys():
            self.memory = LiteLLMMemory(name=f"{self.name}_memory",
                                        is_traced=False)
        else:
            self.memory = kwargs['memory']

        self.with_memory = with_memory
        self.max_tokens = max_tokens
        self.answer_format_prompt = answer_format_prompt
        self.openai_functions = openai_functions
        self.function_callables = function_callables
        self.example_selector = example_selector

        self.total_cost = 0
        # The context builder needs the available token size from the prompt template
        self.answer_format_prompt_size = count_tokens(answer_format_prompt) if answer_format_prompt is not None else 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    async def run(self, **kwargs):
        content = kwargs['content']
        model = kwargs['model'] if kwargs['model'] is not None else self.model
        temperature = kwargs['temperature'] if kwargs['temperature'] is not None else self.temperature
        max_tokens = kwargs['max_tokens'] if kwargs['max_tokens'] is not None else self.max_tokens
        call_metadata = kwargs['call_metadata'] if kwargs['call_metadata'] is not None else {}
        messages = await self.generate_messages_history(role='user',
                                                        content=content)
        with_functions = kwargs['with_functions'] if kwargs['with_functions'] is not None else False
        for key in ['content', 'model', 'temperature', 'max_tokens', 'call_metadata', 'with_functions']: kwargs.pop(key)

        api_result = await self.get_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            n=self.n,
            max_tokens=max_tokens,
            call_metadata=call_metadata,
            with_functions=with_functions,
            **kwargs
        )
        return {
            "response": api_result,
        }

    async def process_output(self, **kwargs):

        # Case if OpenAI decides function call
        if kwargs['response']['choices'][0]['finish_reason'] == 'function_call':
            # Call the function
            self.llm_trace.create_span(
                name=f"Calling function: {kwargs['response']['choices'][0]['message']['function_call']['name']}",
                startTime=datetime.now(),
                metadata={'api_result': kwargs['response']['choices'][0]},
            )
            # Call the function
            function_name = kwargs['response']['choices'][0]['message']['function_call']['name']
            function_result = await self.run_agent_function(
                function_call_info=kwargs['response']['choices'][0]['message']['function_call']
            )
            # Append function result to memory
            function_msg = get_function_message(
                content=function_result,
                name=function_name
            )
            # Generate input messages with the function call content
            messages = await self.generate_messages_history(role='function',
                                                            name=function_name,
                                                            content=function_msg['content'])

            # Make API call with the function call content
            # Remove functions arg to get final assistant response
            api_result = await self.get_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.n,
            )
            assistant_response = api_result['choices'][0]['message']['content']
        else:
            # If no function call, just return the result
            assistant_response = kwargs['response']['choices'][0]['message']['content']
            function_msg = get_assistant_message(content=assistant_response)
            await self.add_memory(new_memory=function_msg)

        return {'response': assistant_response}

    async def get_completion(self,
                             model,
                             temperature,
                             n,
                             max_tokens,
                             messages,
                             call_metadata={},
                             generation_name="Assistant response",
                             with_functions=False,
                             **kwargs):
        try:
            self.llm_trace.create_generation(
                name=generation_name,
                model=model,
                prompt=messages,
                startTime=datetime.now(),
            )

            if with_functions:
                kwargs['functions'] = self.openai_functions
                kwargs['function_call'] = 'auto'

            api_result = await acompletion(
                model=model,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs
            )
            model_parameters = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
            }
            self._update_generation(api_result=api_result.model_dump(),
                                    model_parameters=model_parameters,
                                    call_metadata=call_metadata)
            return api_result.model_dump()

        except Exception as e:
            # cost = get_openai_api_cost(model=self.model,
            #                           completion_tokens=0,
            #                           prompt_tokens=count_openai_messages_tokens(messages))
            self.llm_trace.update_generation(
                completion=str({"error": str(e)}),
                metadata={"error": str(e)},
            )
            raise e

    def _update_generation(self,
                           api_result,
                           model_parameters,
                           call_metadata
                           ):
        # We remove message content to properly visualise the API result in metadata
        # api_result_to_log = api_result.copy()
        # api_result_to_log['choices'][0]['message']['content'] = "..."
        dict_to_log = copy.deepcopy(api_result['choices'][0])
        dict_to_log['message']['content'] = "..."

        # Enrich the api result with metadata
        call_metadata['api_result'] = dict_to_log
        call_metadata['cost_summary'] = get_openai_api_cost(model=self.model,
                                                            completion_tokens=api_result["usage"]['completion_tokens'],
                                                            prompt_tokens=api_result["usage"]['prompt_tokens'])
        # call_metadata['api_result'] = api_result_to_log
        call_metadata['cost_summary']['total_cost'] = self.total_cost

        # Extract completion from API result
        if api_result['choices'][0]['finish_reason'] == 'function_call':
            completion = str(api_result['choices'][0]['message']['function_call'])
        else:
            completion = str(api_result['choices'][0]['message']['content'])

        self.llm_trace.update_generation(
            endTime=datetime.now(),
            modelParameters=model_parameters,
            completion=completion,
            metadata=call_metadata,
            usage=Usage(promptTokens=call_metadata['cost_summary']['prompt_tokens'],
                        completionTokens=call_metadata['cost_summary']['completion_tokens'])
        )
        self.total_cost += call_metadata['cost_summary']['request_cost']

    async def generate_messages_history(self,
                                        role,
                                        content,
                                        **kwargs):
        system_prompt = get_system_message(content=self.system_prompt)
        examples = []  # add example selector
        if self.example_selector and role == 'user':
            best_examples = await self.example_selector(input=content)
            for good_example in best_examples['output']['best_examples']:
                examples.append(get_user_message(good_example['USER']))
                examples.append(get_assistant_message(str(good_example['ASSISTANT'])))

        new_msg = get_openai_message(role,
                                     content,
                                     **kwargs)
        messages = [system_prompt] \
                   + self.memory.get_memories() + \
                   examples + \
                   [new_msg]

        # Add to memory
        memories = await self.memory(role=role,
                                     content=content,
                                     **kwargs)

        return messages

    @property
    def available_token_size(self):
        memories_size = count_tokens(self.memory.memories,
                                     header='',
                                     ignore_keys=[])

        return (OPENAI_MODELS_CONTEXT_SIZES[
                    self.model] - self.prompt_template.size - memories_size - self.max_tokens - self.answer_format_prompt_size - 10) * 0.99

    async def run_agent_function(self,
                                 function_call_info):
        start_time = datetime.now()
        callable = self.function_callables[function_call_info['name']]
        function_args = json.loads(function_call_info['arguments'])

        self.llm_trace.update_span(
            name=f"Running function : {function_call_info['name']}",
            startTime=start_time,
            input=function_args,
        )

        function_result = callable(**function_args)
        self.llm_trace.update_span(endTime=datetime.now(), output={'output': function_result})
        return function_result
