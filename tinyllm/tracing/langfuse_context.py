import datetime as dt
import functools
import inspect
import traceback
from contextlib import asynccontextmanager

import langfuse
import numpy as np

from tinyllm import langfuse_client
from tinyllm.util.helpers import count_tokens

import contextvars
from contextlib import asynccontextmanager

model_parameters = [
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


class LangfuseContext:
    _current_trace = contextvars.ContextVar('current_trace', default=None)
    _current_observation = contextvars.ContextVar('current_observation', default=None)

    @classmethod
    @asynccontextmanager
    async def trace_context(cls, name):
        existing_trace = cls._current_trace.get()
        new_trace_created = False

        if existing_trace is None:
            new_trace = langfuse_client.trace(name=name, userId="test")
            token = cls._current_trace.set(new_trace)
            new_trace_created = True
        else:
            token = None
            new_trace = existing_trace

        try:
            yield new_trace
        finally:
            if new_trace_created:
                cls._current_trace.reset(token)
            langfuse_client.flush()

    @classmethod
    def get_trace(cls):
        return cls._current_trace.get()

    @classmethod
    def get_obs(cls):
        return cls._current_observation.get() or cls.get_trace()

    @classmethod
    def set_obs(cls, observation):
        cls._current_observation.set(observation)

    @classmethod
    def set_parent_obs(cls, observation):
        cls._current_trace.set(observation)


def get_context_observation(type, name, observation_input):
    current_observation = LangfuseContext.get_obs()
    if isinstance(current_observation, langfuse.client.StatefulTraceClient):
        observation_method = getattr(current_observation, type)
        current_observation = observation_method(name=name, **observation_input)
        LangfuseContext.set_parent_obs(current_observation)
        return current_observation
    else:
        observation_method = getattr(current_observation, type)
        obs = observation_method(name=name, **observation_input)
        LangfuseContext.set_obs(obs)
        return obs


def handle_exception(obs, e):
    if 'end' in dir(obs):
        obs.end(level='ERROR', status_message=str(traceback.format_exception(e)))
    elif 'update' in dir(obs):
        obs.update(level='ERROR', status_message=str(traceback.format_exception(e)))
    raise e


def prepare_observation_input(input_mapping, function_kwargs):
    if not input_mapping:
        return {'input': convert_dict_to_string(function_kwargs)}
    else:
        return {langfuse_kwarg: function_kwargs[function_kwarg] for langfuse_kwarg, function_kwarg in
                input_mapping.items()}


def convert_dict_to_string(d):
    for key, value in d.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            convert_dict_to_string(value)
        elif not type(value) in (str, dict, list, tuple, float, int, bool, np.array):
            d[key] = str(value)
    return d


def end_observation(func, obs, observation_input, function_output, output_mapping, observation_type, function_kwargs):
    mapped_obs_output = {}
    if type(function_output) == list:
        for i, item in enumerate(function_output):
            if type(item) == dict:
                convert_dict_to_string(item)
    elif type(function_output) == dict:
        convert_dict_to_string(function_output)

    if not output_mapping:
        mapped_obs_output = {'output': function_output}
    else:
        for langfuse_kwarg, function_kwarg in output_mapping.items():
            mapped_obs_output[langfuse_kwarg] = function_output[function_kwarg]

    if observation_type == 'generation':
        prompt_tokens = count_tokens(observation_input['input'])
        completion_tokens = count_tokens(function_output['message'])
        if inspect.isasyncgenfunction(func):
            response = function_output['last_chunk']
        else:
            response = function_output['response']

        total_tokens = prompt_tokens + completion_tokens

        obs.end(
            end_time=dt.datetime.now(),
            metadata={
                'response': response,
                'run_arguments': function_kwargs
            },
            model_parameters={k: v for k, v in function_kwargs.items() if
                              k in model_parameters},
            usage={
                'promptTokens': prompt_tokens,
                'completionTokens': completion_tokens,
                'totalTokens': total_tokens
            },
            **mapped_obs_output
        )


    elif observation_type == 'span':
        obs.end(**mapped_obs_output)
    elif observation_type == 'trace':
        obs.end(**mapped_obs_output)


async def perform_evaluations(observation, result, evaluators):
    result.update({'observation': observation})
    if evaluators:
        for evaluator in evaluators:
            await evaluator(**result)
    result.pop('observation')


def conditional_args(observation_type, input_mapping=None, output_mapping=None):
    if observation_type == 'generation':
        if input_mapping is None:
            input_mapping = {'input': 'messages'}
        if output_mapping is None:
            output_mapping = {'output': 'message'}
    return input_mapping, output_mapping


def get_obs_name(*args, func):
    name = None
    if len(args) > 0:
        name = args[0].name
    else:
        if hasattr(func, '__qualname__'):
            if func.__qualname__.split('.')[-2] != '<locals>':
                name = '.'.join(func.__qualname__.split('.')[-2::])
            else:
                name = func.__name__
    return name


####### DECORATORS #######


def observation(observation_type, name=None, input_mapping=None, output_mapping=None, evaluators=None):
    input_mapping, output_mapping = conditional_args(observation_type, input_mapping, output_mapping)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **function_kwargs):
            nonlocal name
            if not name:
                name = get_obs_name(*args, func=func)
            observation_input = prepare_observation_input(input_mapping, function_kwargs)
            async with LangfuseContext.trace_context(name):
                obs = get_context_observation(observation_type, name, observation_input)
                # Pass the observation to the Function to run its Evaluators
                if len(args) > 0:
                    args[0].current_observation = obs

                try:
                    result = await func(*args, **function_kwargs)
                    if type(result) != dict:
                        raise Exception('Functions with @observation  must return a dict')
                    end_observation(func, obs, observation_input, result, output_mapping, observation_type,
                                    function_kwargs)

                    # Specific decorator's evaluators
                    await perform_evaluations(obs, result, evaluators)

                    return result
                except Exception as e:
                    handle_exception(obs, e)
                finally:
                    if LangfuseContext.get_obs() == obs:
                        LangfuseContext.set_obs(None)

        return wrapper

    return decorator


def streaming_observation(observation_type, name=None, input_mapping=None, output_mapping=None, evaluators=None):
    input_mapping, output_mapping = conditional_args(observation_type, input_mapping, output_mapping)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **function_kwargs):

            nonlocal name
            if not name:
                name = func.__qualname__ if hasattr(func, '__qualname__') else func.__name__

            observation_input = {}
            if not input_mapping:
                observation_input = {'input': function_kwargs}
            else:
                for langfuse_kwarg, function_kwarg in input_mapping.items():
                    observation_input[langfuse_kwarg] = function_kwargs[function_kwarg]

            async with LangfuseContext.trace_context(name):
                obs = get_context_observation(observation_type, name, observation_input)
                try:
                    async for message in func(*args, **function_kwargs):
                        yield message
                    end_observation(func, obs, observation_input, message, output_mapping, observation_type,
                                    function_kwargs)
                    await perform_evaluations(obs, message, evaluators)
                except Exception as e:
                    handle_exception(obs, e)
                    raise e
                finally:
                    LangfuseContext.set_obs(None)

        return wrapper

    return decorator
