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
        messages = await self.prepare_messages(message=kwargs['message'])
        kwargs['messages'] = messages
        async for msg in self.get_completion(**kwargs):
            yield msg

        # Memorize interaction
        await self.memorize(message=kwargs['message'])
        if msg['type'] == 'tool':
            openai_msg = get_openai_message(role='function', content=msg['completion'], name=msg['completion']['name'])
            await self.memorize(message=openai_msg)
        if msg['type'] == 'completion':
            openai_msg = get_openai_message(role='assistant', content=msg['completion'])
            await self.memorize(message=openai_msg)

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
        with_tools = 'tool_choice' in kwargs and 'tools' in kwargs
        tools_kwargs = {}
        if with_tools: tools_kwargs = {'tools': kwargs['tools'],
                                       'tool_choice': kwargs['tool_choice']}
        response = await acompletion(
            model=model,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            messages=messages,
            stream=True,
            **tools_kwargs
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

                elif chunk_type == "tool":
                    if function_call['name'] is None:
                        function_call['name'] = delta['tool_calls'][0]['function']['name']

                    completion = function_call
                    function_call['arguments'] += delta['tool_calls'][0]['function']['arguments']


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
            return "tool"

        return "completion"

    def get_stream_status(self,
                          chunk):
        if chunk['choices'][0]['finish_reason']:
            return "success"
        else:
            return "streaming"