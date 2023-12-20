import functools
from contextlib import asynccontextmanager

from tinyllm import langfuse_client


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


def observation(type, name=None, input_mapping=None, output_mapping=None, evaluators=None):
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
                current_observation = LangfuseContext.get_current_observation()
                observation_method = getattr(current_observation, type)
                obs = observation_method(name=name, **function_input)
                LangfuseContext.set_current_observation(obs)

                try:
                    result = await func(*args, **kwargs)

                    function_output = {}
                    if not output_mapping:
                        function_output = {'output': result}
                    else:
                        for langfuse_kwarg, function_kwarg in output_mapping.items():
                            function_output[langfuse_kwarg] = result[function_kwarg]

                    if type in ['span', 'generation']:
                        obs.end(**function_output)
                    else:
                        obs.update(**function_output)

                    # Evaluate
                    function_output.update({'observation': obs})

                    if hasattr(func, '__qualname__'):
                        self = args[0]
                        if func.__qualname__.split('.')[-1] == 'run':
                            if getattr(self, 'run_evaluators', None):
                                for evaluator in self.run_evaluators:
                                    await evaluator(**function_output)
                        elif func.__qualname__.split('.')[-1] == 'process_output':
                            if getattr(self, 'process_output_evaluators', None):
                                for evaluator in self.process_output_evaluators:
                                    await evaluator(**function_output, )

                    if evaluators:
                        for evaluator in evaluators:
                            await evaluator(**function_output)

                    return result
                except Exception as e:
                    obs.level = 'ERROR'
                    obs.status_message = str(e)
                    raise
                finally:
                    LangfuseContext.set_current_observation(None)

        return wrapper

    return decorator
