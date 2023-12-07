import datetime as dt
import json

from langfuse.model import CreateGeneration, UpdateGeneration
from litellm import acompletion
from openai import OpenAIError
from tenacity import stop_after_attempt, wait_random_exponential, retry_if_exception_type, retry

from tinyllm.functions.lite_llm.lite_llm import LiteLLM
from tinyllm.function_stream import FunctionStream
from tinyllm.functions.util.helpers import get_openai_message


class LiteLLMStream(LiteLLM, FunctionStream):

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    async def run(self, **kwargs):
        message = get_openai_message(role=kwargs['role'],
                                     content=kwargs['content'])
        messages = await self.process_memory(message)
        kwargs['messages'] = messages
        async for msg in self.get_completion(**kwargs):
            yield msg

    async def get_completion(self,
                             **kwargs):
        self.generation = self.trace.generation(CreateGeneration(
            name='stream: ' + self.name,
            startTime=dt.datetime.now(),
            prompt=kwargs['messages'],
        ))
        kwargs.pop('content')
        response = await acompletion(**kwargs)
        function_call = {
            "name": None,
            "arguments": ""
        }
        # OpenAI function call works as follows: function name available at delta.tool_calls[0].function.
        # It returns a diction where: 'name' is returned only in the first chunk
        # arguments are sent in chunks so need to keep track of them
        async for chunk in response:

            status = self.get_stream_status(chunk)
            chunk_type = self.get_chunk_type(chunk)
            delta = chunk['choices'][0]['delta'].model_dump()

            if status == "streaming":
                if chunk_type == "completion":
                    completion += delta['content']

                elif chunk_type == "tool":
                    if function_call['name'] is None:
                        function_call['name'] = delta['tool_calls'][0]['function']['name']
                    function_call['arguments'] += delta['tool_calls'][0]['function']['arguments']
                    completion = function_call

            elif status == "success":
                if chunk_type == 'tool':
                    function_call['arguments'] = json.loads(function_call['arguments'])

            yield {
                "streaming_status": status,
                "type": chunk_type,
                "delta": delta,
                "completion": completion
            }

        self.generation.update(UpdateGeneration(
            completion=completion,
            endTime=dt.datetime.now(),
        ))

    def get_chunk_type(self,
                       chunk):
        chunk = chunk.model_dump()
        delta = chunk['choices'][0]['delta']

        if 'tool_calls' in delta or chunk['choices'][0]['finish_reason'] == 'tool_calls':
            return "tool"

        delta = chunk['choices'][0]['delta'].model_dump()

        if 'tool_calls' in delta:
            print('hi')

        return "completion"

    def get_stream_status(self,
                          chunk):
        if chunk.model_dump()['choices'][0]['finish_reason']:
            return "success"
        else:
            return "streaming"



        """

    def handle_stream_chunks(self,
                             chunk,
                             completion,
                             function_call: dict):
        # Function call
        if getattr(chunk['choices'][0]['delta'], 'tool_calls', None):
            completion += chunk.choices[0].delta.tool_calls[0].function.arguments
            function_call['arguments'] += chunk.choices[0].delta.tool_calls[0].function.arguments
            # Get function name
            if getattr(chunk.choices[0].delta, 'tool_calls', None):
                if function_call['name'] is None:
                    function_call['name'] = chunk.choices[0].delta.tool_calls[0].function.name

        # Assistant response
        elif isinstance(getattr(chunk['choices'][0]['delta'], 'content', None), str):
            completion += chunk['choices'][0]['delta'].content

        return chunk, completion, function_call


        # Detect if function call
        if getattr(delta, 'tool_calls', None):
            if function_call['name'] is None:
                function_call['name'] = delta.tool_calls[0].function.name
            function_call['arguments'] += delta.tool_calls[0].function.arguments
            completion = function_call
        if last_role == 'assistant' and delta.content:
            completion += delta.content
        """
