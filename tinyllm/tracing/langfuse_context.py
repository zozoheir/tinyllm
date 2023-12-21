import datetime as dt
import functools
import inspect
import traceback
from contextlib import asynccontextmanager

import langfuse

from tinyllm import langfuse_client
from tinyllm.util.helpers import count_tokens

import contextvars
from contextlib import asynccontextmanager


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


def get_context_obs(type, name, function_input):
    current_observation = LangfuseContext.get_obs()
    if isinstance(current_observation, langfuse.client.StatefulTraceClient):
        observation_method = getattr(current_observation, type)
        current_observation = observation_method(name=name, **function_input)
        LangfuseContext.set_parent_obs(current_observation)
        return current_observation
    else:
        observation_method = getattr(current_observation, type)
        obs = observation_method(name=name, **function_input)
        LangfuseContext.set_obs(obs)
        return obs


def handle_exception(obs, e):
    if 'end' in dir(obs):
        obs.end(level='ERROR', status_message=str(traceback.format_exception(e)))
    elif 'update' in dir(obs):
        obs.update(level='ERROR', status_message=str(traceback.format_exception(e)))
    raise e


def prepare_observation_input(input_mapping, kwargs):
    if not input_mapping:
        # stringify all non string or dict values
        for key, value in kwargs.items():
            if not isinstance(value, (str, dict)):
                kwargs[key] = str(value)
        return {'input': kwargs}
    return {langfuse_kwarg: kwargs[function_kwarg] for langfuse_kwarg, function_kwarg in input_mapping.items()}


def convert_dict_to_string(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_dict_to_string(value)
        elif not isinstance(value, (str, dict, list, tuple)):
            d[key] = str(value)


def end_observation(obs, function_input, function_output, output_mapping, observation_type):
    mapped_output = {
        # 'end_time':dt.datetime.utcnow(),
    }
    if type(function_output) == list:
        for i, item in enumerate(function_output):
            if type(item) == dict:
                convert_dict_to_string(item)
    elif type(function_output) == dict:
        convert_dict_to_string(function_output)

    if not output_mapping:
        mapped_output = {'output': function_output}
    else:
        for langfuse_kwarg, function_kwarg in output_mapping.items():
            mapped_output[langfuse_kwarg] = function_output[function_kwarg]

    if observation_type == 'generation':
        prompt_tokens = count_tokens(function_input)
        completion_tokens = count_tokens(function_output)
        total_tokens = prompt_tokens + completion_tokens

        obs.end(
            end_time=dt.datetime.now(),
            usage={
                'promptTokens': prompt_tokens,
                'completionTokens': completion_tokens,
                'totalTokens': total_tokens,
            },
            **mapped_output)
    elif observation_type == 'span':
        obs.end(**mapped_output)
    elif observation_type == 'trace':
        obs.end(**mapped_output)


async def perform_evaluations(observation, result, func, args, evaluators):
    if inspect.ismethod(func):
        result.update({'observation': observation})
        self = args[0]

        # Function.run and Function.process_output evaluators
        if func.__qualname__.split('.')[-1] == 'run':
            if getattr(self, 'run_evaluators', None):
                for evaluator in self.run_evaluators:
                    await evaluator(**result)
        elif func.__qualname__.split('.')[-1] == 'process_output':
            if getattr(self, 'process_output_evaluators', None):
                for evaluator in self.process_output_evaluators:
                    await evaluator(**result, )

        if evaluators:
            for evaluator in evaluators:
                await evaluator(**result)


def conditional_args(observation_type, input_mapping=None, output_mapping=None):
    if observation_type == 'generation':
        if input_mapping is None:
            input_mapping = {'input': 'messages'}
        if output_mapping is None:
            output_mapping = {'output': 'message'}
    return input_mapping, output_mapping


####### DECORATORS #######

def observation(observation_type, name=None, input_mapping=None, output_mapping=None, evaluators=None):
    input_mapping, output_mapping = conditional_args(observation_type, input_mapping, output_mapping)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal name
            if not name:
                name = func.__qualname__ if hasattr(func, '__qualname__') else func.__name__
            function_input = prepare_observation_input(input_mapping, kwargs)
            function_input['start_time'] = dt.datetime.utcnow()
            async with LangfuseContext.trace_context(name):
                obs = get_context_obs(observation_type, name, function_input)
                try:
                    result = await func(*args, **kwargs)
                    end_observation(obs, function_input, result, output_mapping, observation_type)
                    await perform_evaluations(obs, result, func, args, evaluators)
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
        async def wrapper(*args, **kwargs):

            nonlocal name
            if not name:
                name = func.__qualname__ if hasattr(func, '__qualname__') else func.__name__

            function_input = {}
            if not input_mapping:
                function_input = {'input': kwargs}
            else:
                for langfuse_kwarg, function_kwarg in input_mapping.items():
                    function_input[langfuse_kwarg] = kwargs[function_kwarg]

            async with LangfuseContext.trace_context(name):
                obs = get_context_obs(observation_type, name, function_input)
                try:
                    async for message in func(*args, **kwargs):
                        yield message
                    end_observation(obs, function_input, message, output_mapping, observation_type)
                    await perform_evaluations(obs, message, func, args, evaluators)
                except Exception as e:
                    handle_exception(obs, e)
                    raise e
                finally:
                    LangfuseContext.set_obs(None)

        return wrapper

    return decorator
