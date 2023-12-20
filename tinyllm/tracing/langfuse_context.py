import functools
import inspect
import traceback
from contextlib import asynccontextmanager

from tinyllm import langfuse_client
from tinyllm.util.helpers import count_tokens


class LangfuseContext:
    _current_trace = None
    _current_observation = None

    @classmethod
    @asynccontextmanager
    async def trace_context(cls, name):
        if cls._current_trace is None:
            # Create a new trace if there isn't an existing one
            cls._current_trace = langfuse_client.trace(name=name, userId="test")
            new_trace_created = True
        else:
            # Use the existing trace
            new_trace_created = False

        try:
            yield cls._current_trace
        finally:
            if new_trace_created:
                cls._current_trace = None

    @classmethod
    def get_current_observation(cls):
        return cls._current_observation or cls._current_trace

    @classmethod
    def set_current_observation(cls, observation):
        cls._current_observation = observation


def handle_exception(obs, e):
    obs.level = 'ERROR'
    obs.status_message = str(e)
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


def update_observation(obs, function_input, function_output, output_mapping, type):
    mapped_output = {}
    for key, value in function_output.items():
        if not isinstance(value, (str, dict)):
            function_output[key] = str(value)

    if not output_mapping:
        mapped_output = {'output': function_output}
    else:
        for langfuse_kwarg, function_kwarg in output_mapping.items():
            mapped_output[langfuse_kwarg] = function_output[function_kwarg]


    if type == 'generation':
        prompt_tokens = count_tokens(function_input)
        completion_tokens = count_tokens(function_output)
        total_tokens = prompt_tokens + completion_tokens
        obs.end(usage={
            'promptTokens': prompt_tokens,
            'completionTokens': completion_tokens,
            'totalTokens': total_tokens,
        },
            **mapped_output)
    elif type == 'span':
        obs.end(**mapped_output)
    else:
        obs.update(**mapped_output)


async def perform_evaluations(observation, result, func, args, evaluators):
    result.update({'observation': observation})
    if inspect.ismethod(func):
        self = args[0]
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


def get_context_obs(type, name, function_input):
    current_observation = LangfuseContext.get_current_observation()
    observation_method = getattr(current_observation, type)
    obs = observation_method(name=name, **function_input)
    LangfuseContext.set_current_observation(obs)
    return obs


def conditional_args(type, input_mapping=None, output_mapping=None):
    if type == 'generation':
        if input_mapping is None:
            input_mapping = {'input': 'messages'}
        if output_mapping is None:
            output_mapping = {'output': 'message'}
    return input_mapping, output_mapping


####### DECORATORS #######

def observation(type, name=None, input_mapping=None, output_mapping=None, evaluators=None):
    input_mapping, output_mapping = conditional_args(type, input_mapping, output_mapping)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal name
            if not name:
                name = func.__qualname__ if hasattr(func, '__qualname__') else func.__name__
            function_input = prepare_observation_input(input_mapping, kwargs)

            async with LangfuseContext.trace_context(name):
                obs = get_context_obs(type, name, function_input)
                try:
                    result = await func(*args, **kwargs)
                    update_observation(obs, function_input, result, output_mapping, type)
                    # Evaluate
                    await perform_evaluations(obs, result, func, args, evaluators)
                    return result
                except Exception as e:
                    handle_exception(obs, e)
                    raise e
                finally:
                    LangfuseContext.set_current_observation(None)

        return wrapper

    return decorator


def streaming_observation(type, name=None, input_mapping=None, output_mapping=None, evaluators=None):
    input_mapping, output_mapping = conditional_args(type, input_mapping, output_mapping)

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
                obs = get_context_obs(type, name, function_input)
                try:
                    async for message in func(*args, **kwargs):
                        yield message
                    update_observation(obs, function_input, message, output_mapping, type)
                    await perform_evaluations(obs, message, func, args, evaluators)
                except Exception as e:
                    handle_exception(obs, e)
                    raise e
                finally:
                    LangfuseContext.set_current_observation(None)

        return wrapper

    return decorator
