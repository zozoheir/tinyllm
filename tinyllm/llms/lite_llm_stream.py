import datetime as dt

from langfuse.model import CreateGeneration, UpdateGeneration, Usage
from litellm import acompletion
from openai import OpenAIError
from tenacity import stop_after_attempt, wait_random_exponential, retry_if_exception_type, retry

from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.function_stream import FunctionStream
from tinyllm.tracing.generation import langfuse_generation_stream
from tinyllm.util.helpers import count_tokens


class LiteLLMStream(LiteLLM, FunctionStream):

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError))
    )
    @langfuse_generation_stream
    async def run(self, **kwargs):
        tools_args = {}
        if kwargs.get('tools', None) is not None:
            tools_args = {'tools': kwargs.get('tools', None),
                          'tool_choice': kwargs.get('tool_choice', 'auto')}

        response = await acompletion(
            model=kwargs.get('model', 'gpt-3.5-turbo'),
            temperature=kwargs.get('temperature', 0),
            n=kwargs.get('n', 1),
            max_tokens=kwargs.get('max_tokens', 400),
            messages=kwargs['messages'],
            stream=True,
            **tools_args
        )

        # We need to track 2 things: the response delta and the function_call
        delta_to_return = None
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

                elif chunk_type == "tool":
                    if function_call['name'] is None:
                        function_call['name'] = delta['tool_calls'][0]['function']['name']
                    if delta_to_return is None:
                        delta_to_return = delta

                    completion = function_call
                    function_call['arguments'] += delta['tool_calls'][0]['function']['arguments']

            yield {
                "streaming_status": status,
                "type": chunk_type,
                "delta": delta_to_return or '',
                "completion": completion
            }

    def get_chunk_type(self,
                       chunk):
        delta = chunk['choices'][0]['delta']

        if delta.get('tool_calls', None) is not None or chunk['choices'][0]['finish_reason'] == 'tool_calls':
            return "tool"

        return "completion"

    def get_stream_status(self,
                          chunk):
        if chunk['choices'][0]['finish_reason']:
            return "success"
        else:
            return "streaming"
