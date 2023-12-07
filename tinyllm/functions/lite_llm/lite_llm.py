import json
import os
from datetime import datetime
from typing import Optional, Any
import copy

import openai
from langfuse.model import Usage, CreateGeneration, UpdateGeneration
from litellm import OpenAIError, acompletion

from tinyllm.function import Function
from tinyllm.functions.lite_llm.lite_llm_memory import Memory
from tinyllm.functions.util.example_selector import ExampleSelector
from tinyllm.functions.util.helpers import *
from tinyllm.validator import Validator
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
    system_prompt: str = "You are a helpful assistant"
    with_memory: bool = False
    answer_format_prompt: Optional[str]
    example_selector: Optional[ExampleSelector]


class LiteLLMChatInputValidator(Validator):
    role: str
    content: str
    model: Optional[str] = 'gpt-3.5-turbo'
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 400
    stream: Optional[bool] = True


class LiteLLMChatOutputValidator(Validator):
    response: Any


class LiteLLM(Function):
    def __init__(self,
                 system_prompt="You are a helpful assistant",
                 with_memory=False,
                 answer_format_prompt=None,
                 example_selector=None,
                 **kwargs):
        val = LiteLLMChatInitValidator(system_prompt=system_prompt,
                                       with_memory=with_memory,
                                       answer_format_prompt=answer_format_prompt,
                                       example_selector=example_selector,
                                       )
        super().__init__(input_validator=LiteLLMChatInputValidator,
                         output_validator=LiteLLMChatOutputValidator,
                         **kwargs)
        self.system_prompt = system_prompt
        self.n = 1
        if 'memory' not in kwargs.keys():
            self.memory = Memory(name=f"{self.name}_memory",
                                 is_traced=self.is_traced,
                                 debug=self.debug,
                                 trace=self.trace)
        else:
            self.memory = kwargs['memory']

        self.with_memory = with_memory
        self.answer_format_prompt = answer_format_prompt
        self.example_selector = example_selector

        self.total_cost = 0
        # The context builder needs the available token size from the prompt template
        self.answer_format_prompt_size = count_tokens(answer_format_prompt) if answer_format_prompt is not None else 0
        self.completion_args = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    async def run(self, **kwargs):
        message = kwargs['message']
        api_result = await self.get_completion(
            **kwargs
        )
        return {
            "response": api_result,
        }

    async def get_completion(self,
                             model,
                             temperature,
                             n,
                             max_tokens,
                             messages,
                             call_metadata={},
                             generation_name="Assistant response",
                             **kwargs):
        try:
            self.trace.create_generation(
                name=generation_name,
                model=model,
                prompt=messages,
                startTime=datetime.now(),
            )

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
            self.trace.update_generation(
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

        self.trace.update_generation(
            endTime=datetime.now(),
            modelParameters=model_parameters,
            completion=completion,
            metadata=call_metadata,
            usage=Usage(promptTokens=call_metadata['cost_summary']['prompt_tokens'],
                        completionTokens=call_metadata['cost_summary']['completion_tokens'])
        )
        self.total_cost += call_metadata['cost_summary']['request_cost']

    async def generate_messages_history(self,
                                        message):
        system_prompt = get_system_message(content=self.system_prompt)
        examples = []  # add example selector
        if self.example_selector and message['role'] == 'user':
            best_examples = await self.example_selector(input=message['content'])
            for good_example in best_examples['output']['best_examples']:
                examples.append(get_user_message(good_example['USER']))
                examples.append(get_assistant_message(str(good_example['ASSISTANT'])))

        messages = [system_prompt] \
                   + self.memory.get_memories() + \
                   examples + \
                   [message]

        # Add to memory
        memories = await self.memory(message=message)

        return messages

    @property
    def available_token_size(self):
        memories_size = count_tokens(self.memory.memories,
                                     header='',
                                     ignore_keys=[])

        return (OPENAI_MODELS_CONTEXT_SIZES[
                    self.model] - self.prompt_template.size - memories_size - self.max_tokens - self.answer_format_prompt_size - 10) * 0.99

    async def run_agent_function(self,
                                 function_call_message: dict):
        start_time = datetime.now()
        callable = self.tools_callables[function_call_message['name']]
        function_args = json.loads(function_call_message['arguments'])
        self.trace.update_span(
            name=f"Running function : {function_call_message['name']}",
            startTime=start_time,
            input=function_args,
        )
        function_result = callable(**function_args)
        self.trace.update_span(endTime=datetime.now(), output={'output': function_result})
        return function_result


"""


    async def process_output(self, **kwargs):
        # Case if OpenAI decides function call
        if kwargs['chunk_dict']['choices'][0]['finish_reason'] == 'tool_calls':
            # Call the function
            self.trace.create_span(
                name=f"Calling function: {kwargs['assistant_response']['name']}",
                startTime=datetime.now(),
                metadata={'function_call': kwargs},
            )
            # Call the function
            function_name = kwargs['assistant_response']['name']
            function_result = await self.run_agent_function(
                function_call_message=kwargs['assistant_response']
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
            acompletion_args = {
                "model": self.model,
                "temperature": self.temperature,
                "n": self.n,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "stream": True,  # Enable streaming
            }
            response = await acompletion(**acompletion_args)
            assistant_response = ""
            last_role = None
            async for chunk in response:
                if chunk['choices'][0]['delta'].role:
                    if chunk['choices'][0]['delta'].role:
                        last_role = chunk['choices'][0]['delta'].role
                    if last_role == 'assistant' and chunk['choices'][0]['delta'].content:
                        assistant_response += chunk['choices'][0]['delta'].content
                        yield chunk, assistant_response

        elif kwargs['chunk_dict']['choices'][0]['finish_reason'] == 'stop':
            # If no function call, just return the result
            assistant_response = kwargs['assistant_response']
            msg = get_openai_message(role='assistant',
                                     content=assistant_response)
            await self.memory(**msg)

"""
