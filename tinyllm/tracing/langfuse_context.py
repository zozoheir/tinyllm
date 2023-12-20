import functools
import uuid
from contextlib import contextmanager

from tinyllm import langfuse_client

class LangfuseIntegration:
    _current_trace = None
    _current_observation = None

    @classmethod
    @contextmanager
    def trace_context(cls, name):
        if cls._current_trace is None:
            # Only create a new trace if there isn't an existing one
            old_trace = None
            cls._current_trace = langfuse_client.trace(name=name, userId="test")
        else:
            # Use the existing trace
            old_trace = cls._current_trace

        try:
            yield cls._current_trace
        finally:
            if old_trace is None:
                cls._current_trace = None

    @classmethod
    def get_current_observation(cls):
        return cls._current_observation or cls._current_trace

def get_observation_input_output(kwargs, result, input_key=None, output_keys=None):
    input_data = kwargs if input_key is None else kwargs.get(input_key, {})
    output_data = result if output_keys is None else {key: result.get(key) for key in output_keys}
    return input_data, output_data

def langfuse_span(input=None, output=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = getattr(args[0], 'name', func.__name__)
            observation = LangfuseIntegration.get_current_observation()
            span = observation.span(name=name)

            try:
                result = await func(*args, **kwargs)
                input_data, output_data = get_observation_input_output(kwargs, result, input_key=input, output_keys=output)
                span.end(output=output_data)
                return result
            except Exception as e:
                span.update(level='ERROR', status_message=str(e))
                raise
        return wrapper
    return decorator

def langfuse_generation(input=None, output=None, metadata=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = getattr(args[0], 'name', func.__name__)
            observation = LangfuseIntegration.get_current_observation()
            generation = observation.generation(name=name, metadata=kwargs.get(metadata) if metadata else {})

            try:
                result = await func(*args, **kwargs)
                input_data, output_data = get_observation_input_output(kwargs, result, input_key=input, output_keys=output)
                generation.end(output=output_data)
                return result
            except Exception as e:
                generation.update(level='ERROR', status_message=str(e))
                raise
        return wrapper
    return decorator

def langfuse_event(input=None, output=None, metadata=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = getattr(args[0], 'name', func.__name__)
            observation = LangfuseIntegration.get_current_observation()
            event = observation.event(name=name, metadata=kwargs.get(metadata) if metadata else {})

            try:
                result = await func(*args, **kwargs)
                input_data, output_data = get_observation_input_output(kwargs, result, input_key=input, output_keys=output)
                event.output = output_data
                return result
            except Exception as e:
                event.level = 'ERROR'
                event.status_message = str(e)
                raise
        return wrapper
    return decorator

