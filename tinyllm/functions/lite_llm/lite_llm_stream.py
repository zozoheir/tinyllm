import datetime as dt

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
        messages = await self.generate_messages_history(message)
        kwargs['messages'] = messages
        async for msg in self.get_completion(**kwargs):
            yield msg

    """
    # Detect if function call
    if getattr(delta, 'tool_calls', None):
        if function_call['name'] is None:
            function_call['name'] = delta.tool_calls[0].function.name
        function_call['arguments'] += delta.tool_calls[0].function.arguments
        completion = function_call
    if last_role == 'assistant' and delta.content:
        completion += delta.content
    """

    async def get_completion(self,
                             **kwargs):
        self.generation = self.trace.generation(CreateGeneration(
            name='stream: ' + self.name,
            startTime=dt.datetime.now(),
            prompt=kwargs['messages'],
        ))
        for i in ['role', 'content']: kwargs.pop(i)
        response = await acompletion(**kwargs)
        completion = ""
        function_call = {
            "name": None,
            "arguments": ""
        }
        async for chunk in response:
            print(chunk)
            status = self.get_stream_status(chunk)
            chunk_type = self.get_chunk_type(chunk)

            if status == "streaming":

                if chunk_type == "completion":
                    completion = self.handle_completion_streaming(
                        chunk,
                        completion
                    )
                elif chunk_type == "tool":
                    self.handle_function_streaming(
                        chunk,
                        completion,
                        function_call
                    )
                    print('hi')
            elif status == "finished":
                self.handle_finished_streaming(
                    chunk,
                    completion
                )

            yield {
                "status": chunk,
                "completion": completion
            }

        self.generation.update(UpdateGeneration(
            completion=completion,
            endTime=dt.datetime.now(),
        ))

    def get_chunk_type(self,
                       chunk):
        delta = chunk['choices'][0]['delta'].model_dump()

        if 'tool_calls' in delta:
            print('hi')

        return "completion"

    def get_stream_status(self,
                          chunk):

        print('hi')
        return "streaming"

    def handle_function_streaming(self,
                                  chunk,
                                  completion,
                                  function_call):
        return chunk, completion, function_call

    def handle_completion_streaming(self,
                                    chunk,
                                    completion):
        return ""

    def handle_finished_streaming(self,
                                  chunk,
                                  completion):
        return ""
