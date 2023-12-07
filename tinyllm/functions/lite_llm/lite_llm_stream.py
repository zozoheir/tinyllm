import datetime as dt
import json

from langfuse.model import CreateGeneration, UpdateGeneration, Usage
from litellm import acompletion
from openai import OpenAIError
from tenacity import stop_after_attempt, wait_random_exponential, retry_if_exception_type, retry

from tinyllm.functions.lite_llm.lite_llm import LiteLLM
from tinyllm.function_stream import FunctionStream
from tinyllm.functions.util.helpers import get_openai_message, count_tokens


class LiteLLMStream(LiteLLM, FunctionStream):

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    async def run(self, **kwargs):
        message = get_openai_message(role=kwargs['role'],
                                     content=kwargs['content'])
        messages = await self.prepare_messages(message=message)
        kwargs['messages'] = messages
        async for msg in self.get_completion(**kwargs):
            yield msg

        # Memorize interaction
        await self.memorize(role=message['role'],
                            content=message['content'])
        if msg['type'] == 'function':
            await self.memorize(
                role='function',
                content=msg['completion'],
            )
        elif msg['type'] == 'completion':
            await self.memorize(
                role='assistant',
                content=msg['completion'],
            )

    async def get_completion(self,
                             model,
                             temperature,
                             n,
                             max_tokens,
                             messages,
                             **kwargs):
        self.generation = self.trace.generation(CreateGeneration(
            name=self.name,
            startTime=dt.datetime.now(),
            prompt=messages,
        ))
        response = await acompletion(
            model=model,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            messages=messages,
            stream=True,
        )
        function_call = {
            "name": None,
            "arguments": ""
        }
        completion = ""
        # OpenAI function call works as follows: function name available at delta.tool_calls[0].function.
        # It returns a diction where: 'name' is returned only in the first chunk
        # tool argument tokens are sent in chunks after so need to keep track of them
        async for chunk in response:
            chunk_dict = chunk.model_dump()
            status = self.get_stream_status(chunk_dict)
            chunk_type = self.get_chunk_type(chunk_dict)
            delta = chunk_dict['choices'][0]['delta']

            # If streaming , we need to store chunks of the completion/function call
            if status == "streaming":

                if chunk_type == "completion":
                    completion += delta['content']

                elif chunk_type == "function":
                    if function_call['name'] is None:
                        function_call['name'] = delta['tool_calls'][0]['function']['name']
                        function_call['arguments'] += delta['tool_calls'][0]['function']['arguments']
                        completion = function_call

            elif status == "success":
                if chunk_type == 'tool':
                    # JSON parsing of tool arguments
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
            usage=Usage(promptTokens=count_tokens(messages), completionTokens=count_tokens(completion)),
        ))

    def get_chunk_type(self,
                       chunk):
        delta = chunk['choices'][0]['delta']

        if 'tool_calls' in delta or chunk['choices'][0]['finish_reason'] == 'tool_calls':
            return "function"

        return "completion"

    def get_stream_status(self,
                          chunk):
        if chunk['choices'][0]['finish_reason']:
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
