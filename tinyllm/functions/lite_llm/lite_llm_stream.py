import datetime as dt

from langfuse.model import CreateGeneration, UpdateGeneration
from litellm import acompletion
from openai import OpenAIError
from tenacity import stop_after_attempt, wait_random_exponential, retry_if_exception_type, retry

from tinyllm.functions.lite_llm.lite_llm import LiteLLM
from tinyllm.function_stream import FunctionStream


class LiteLLMStream(LiteLLM, FunctionStream):

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    async def run(self, **kwargs):
        message = kwargs.pop('message')
        self.completion_args = kwargs
        messages = await self.generate_messages_history(message)
        kwargs['messages'] = messages
        async for chunk, assistant_response in self.get_completion(**kwargs):
            yield chunk, assistant_response

    async def get_completion(self,
                             **kwargs):
        self.generation = self.trace.generation(CreateGeneration(
            name='stream: '+self.name,
            startTime=dt.datetime.now(),
            prompt=kwargs['messages'],
        ))
        response = await acompletion(**kwargs)
        assistant_response = ""
        function_call = {
            "name": None,
            "arguments": ""
        }
        last_role = None
        async for chunk in response:
            if chunk['choices'][0]['delta'].role:
                last_role = chunk['choices'][0]['delta'].role
            if getattr(chunk.choices[0].delta, 'tool_calls', None):
                if function_call['name'] is None:
                    function_call['name'] = chunk.choices[0].delta.tool_calls[0].function.name
                function_call['arguments'] += chunk.choices[0].delta.tool_calls[0].function.arguments
                assistant_response = function_call
            if last_role == 'assistant' and chunk['choices'][0]['delta'].content:
                assistant_response += chunk['choices'][0]['delta'].content

            yield chunk, assistant_response

        self.generation.update(UpdateGeneration(
            completion=assistant_response,
            endTime=dt.datetime.now(),
        ))
