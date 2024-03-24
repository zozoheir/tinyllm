import openai
from litellm import acompletion
from openai import OpenAIError
from tenacity import stop_after_attempt, wait_random_exponential, retry_if_exception_type, retry

from tinyllm.llms.lite_llm import LiteLLM, DEFAULT_CONTEXT_FALLBACK_DICT, DEFAULT_LLM_MODEL, model_parameters
from tinyllm.function_stream import FunctionStream
from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message


class LiteLLMStream(LiteLLM, FunctionStream):

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError, openai.InternalServerError))
    )
    @observation(observation_type='generation', stream=True)
    async def run(self, **kwargs):
        kwargs['messages'] = self._parse_mesages(kwargs['messages'])
        tools_args = self._validate_tool_args(**kwargs)
        completion_args = {arg: kwargs[arg] for arg in kwargs if arg in model_parameters}
        completion_args.update(tools_args)
        completion_args['stream'] = True

        response = await acompletion(
            **completion_args,
        )

        # We need to track 2 things: the response delta and the function_call
        function_call = {
            "name": None,
            "arguments": ""
        }
        completion = ""
        last_completion_delta = None
        finish_delta = None

        # OpenAI function call works as follows: function name available at delta.tool_calls[0].function.
        # It returns a diction where: 'name' is returned only in the first chunk
        # tool argument tokens are sent in chunks after so need to keep track of them

        async for chunk in response:
            chunk_dict = chunk.dict()
            status = self.get_streaming_status(chunk_dict)
            chunk_role = self.get_chunk_type(chunk_dict)
            delta = chunk_dict['choices'][0]['delta']

            # When using tools:
            # We need the last response delta as it contains the full function message
            # The finish message does not contain any delta so we need to keep track of all deltas

            # If streaming , we need to store chunks of the completion/function call
            if status == "streaming":
                if chunk_role == "assistant":
                    completion += delta['content']
                    last_completion_delta = delta
                elif chunk_role == "tool":
                    if function_call['name'] is None:
                        function_call['name'] = delta['tool_calls'][0]['function']['name']
                    if last_completion_delta is None:
                        last_completion_delta = delta

                    completion = function_call
                    function_call['arguments'] += delta['tool_calls'][0]['function']['arguments']

            elif status == "finished-streaming":
                finish_delta = delta

            yield {
                "streaming_status": status,
                "type": chunk_role,
                "last_completion_delta": last_completion_delta,
                "finish_delta": finish_delta,
                "completion": completion,
                "message": get_openai_message(role=chunk_role,
                                              content=completion),
                "last_chunk": chunk_dict,
            }

    def get_chunk_type(self,
                       chunk):
        delta = chunk['choices'][0]['delta']

        if delta.get('tool_calls', None) is not None or chunk['choices'][0]['finish_reason'] == 'tool_calls':
            return "tool"

        return "assistant"

    def get_streaming_status(self,
                             chunk):
        if chunk['choices'][0]['finish_reason'] in ['stop', 'tool_calls']:
            return "finished-streaming"
        else:
            return "streaming"
