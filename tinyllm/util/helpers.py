from typing import Union, List, Dict

import tiktoken
from langchain_community.callbacks.openai_info import standardize_model_name, MODEL_COST_PER_1K_TOKENS, \
    get_openai_token_cost_for_model

from tinyllm.util.prompt_util import stringify_dict

OPENAI_MODELS_CONTEXT_SIZES = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-16k-0613": 16385,
    "text-ada-001": 2049,
    "ada": 2049,
    "text-babbage-001": 2040,
    "babbage": 2049,
    "text-curie-001": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}


def get_user_message(content):
    return {'role': 'user',
            'content': content}


def get_system_message(content):
    return {'role': 'system',
            'content': content}


def get_function_message(content, name):
    return {'role': 'function',
            'name': name,
            'content': content}


def get_assistant_message(content):
    return {'role': 'assistant',
            'content': content}

def get_openai_message(role,
                       content,
                       **kwargs):
    if role not in ['user', 'system', 'function','tool', 'assistant']:
        raise ValueError(f"Invalid role {role}.")

    msg = {'role': role,
           'content': str(content)}
    msg.update(kwargs)
    return msg

def get_openai_api_cost(model: str,
                        completion_tokens: int,
                        prompt_tokens: int):
    """Collect token usage."""
    total_tokens = completion_tokens + prompt_tokens
    model_name = standardize_model_name(model)
    total_cost = 0
    if model_name in MODEL_COST_PER_1K_TOKENS:
        completion_cost = get_openai_token_cost_for_model(
            model_name, completion_tokens, is_completion=True
        )
        prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
        total_cost += prompt_cost + completion_cost
    return {
        'request_cost': total_cost,
        'total_tokens': total_tokens,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
    }


def num_tokens_from_string(string: str, encoding_name: str = 'cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def count_openai_messages_tokens(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in ["gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002"]:
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return int(num_tokens)
    else:
        raise NotImplementedError("openai_num_tokens_from_messages() is not implemented for this model.")


def count_tokens(input: Union[List[Dict], Dict, str],
                 **kwargs):
    if isinstance(input, list):
        if len(input) == 0:
            return 0
        if isinstance(input[0], str):
            return sum([num_tokens_from_string(string) for string in input])
        return sum([count_tokens(input_dict, **kwargs) for input_dict in input])
    elif isinstance(input, str):
        return num_tokens_from_string(input)
    elif isinstance(input, dict):
        dict_string = stringify_dict(header=kwargs.get('header', '[doc]'),
                                     dict=input,
                                     include_keys=kwargs.get('include_keys', []))
        return num_tokens_from_string(dict_string)

    else:
        raise NotImplementedError("count_tokens() is not implemented for this input type.")

