from typing import Optional, Any

import openai
import litellm
from litellm import acompletion
from openai import OpenAIError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from tinyllm.function import Function
from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import *
from tinyllm.util.message import Content, Message
from tinyllm.validator import Validator

litellm.set_verbose = False

model_parameters = [
    "messages",
    "model",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_p"
]


DEFAULT_LLM_MODEL = 'gpt-3.5-turbo-0125'
json_mode_models = ['gpt-3.5-turbo-1106',
                    'gpt-4-1106-preview',
                    'azure/gpt35turbo1106']
OPENAI_TOKEN_LIMITS = {
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-instruct": 4096,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16385,
    "gpt-3.5-turbo-0301": 4096,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
    "code-davinci-002": 8001,
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview	": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    "gpt-4-0314": 8192,
    "gpt-4-32k-0314": 32768,
}

ANYSCALE_TOKEN_LIMITS = {
    "anyscale/Open-Orca/Mistral-7B-OpenOrca": 8192,
    "anyscale/meta-llama/Llama-2-70b-chat-hf": 4096,
}

AZURE_TOKEN_LIMITS = {
    "azure/gpt41106": 128000,
    "azure/gpt35turbo0125": 16385,
    "azure/gpt35turbo1106": 16385,  # JSON MODE
    "azure/gpt4o0513": 128000,
}

LLM_TOKEN_LIMITS = {**OPENAI_TOKEN_LIMITS, **ANYSCALE_TOKEN_LIMITS, **AZURE_TOKEN_LIMITS}



DEFAULT_CONTEXT_FALLBACK_DICT = {
    "gpt-3.5-turbo-0125": "gpt-4-turbo-preview",
    "gpt-4-1106-preview": "gpt-4-1106-preview",
    "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo-16k",
    "azure/gpt41106": "azure/gpt41106",
    "azure/gpt35turbo0125": "azure/gpt41106",
    "azure/gpt35turbo1106": "azure/gpt41106",
    "anyscale/Open-Orca/Mistral-7B-OpenOrca": "gpt-3.5-turbo-16k",
    "anyscale/meta-llama/Llama-2-70b-chat-hf": "gpt-3.5-turbo-16k",
}


class LiteLLMChatInitValidator(Validator):
    system_role: str
    answer_format_prompt: Optional[str]


class LiteLLMChatInputValidator(Validator):
    messages: List[Union[Dict, Message]]
    model: Optional[str] = 'gpt-3.5-turbo'
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 850
    n: Optional[int] = 1
    stream: Optional[bool] = False
    context_window_fallback_dict: Optional[Dict] = DEFAULT_CONTEXT_FALLBACK_DICT


class LiteLLMChatOutputValidator(Validator):
    type: str
    message: dict
    response: Any


class LiteLLM(Function):
    def __init__(self, **kwargs):
        super().__init__(input_validator=LiteLLMChatInputValidator,
                         **kwargs)
        self.generation = None

    def _validate_tool_args(self, **kwargs):
        tools_args = {}
        if kwargs.get('tools', None) is not None:
            tools_args = {
                'tools': kwargs['tools'],
                'tool_choice': kwargs.get('tool_choice', 'auto')
            }
        return tools_args

    def _parse_mesages(self, messages):
        if isinstance(messages[0], Message):
            messages = [message.to_dict() for message in messages]
        return messages

    @observation(observation_type='generation', input_mapping={'input': 'messages'},
                 output_mapping={'output': 'response'})
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type((OpenAIError, openai.InternalServerError))
    )
    async def run(self, **kwargs):
        kwargs['messages'] = self._parse_mesages(kwargs['messages'])
        tools_args = self._validate_tool_args(**kwargs)
        completion_kwargs = {arg: kwargs[arg] for arg in kwargs if arg in model_parameters}
        completion_kwargs.update(tools_args)
        api_result = await acompletion(
            **completion_kwargs,
        )
        model_dump = api_result.dict()
        msg_type = 'tool' if model_dump['choices'][0]['finish_reason'] == 'tool_calls' else 'completion'
        message = model_dump['choices'][0]['message']
        return {
            "type": msg_type,
            "message": message,
            "response": model_dump,
            "completion": message['content'],
        }
