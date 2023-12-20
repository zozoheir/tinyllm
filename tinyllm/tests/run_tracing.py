import asyncio
import functools
from contextlib import asynccontextmanager

from tinyllm import langfuse_client

class LangfuseIntegration:
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

def observation(type, name=None, input_mapping=None, output_mapping=None):
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

            async with LangfuseIntegration.trace_context(name):
                current_observation = LangfuseIntegration.get_current_observation()
                observation_method = getattr(current_observation, type)
                obs = observation_method(name=name, **function_input)
                LangfuseIntegration.set_current_observation(obs)

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
                    return result
                except Exception as e:
                    obs.level = 'ERROR'
                    obs.status_message = str(e)
                    raise
                finally:
                    LangfuseIntegration.set_current_observation(None)

        return wrapper
    return decorator

# Example usage
@observation(type='span', name="search", input_mapping={'input': 'search_query'}, output_mapping={'output': 'docs', 'metadata': 'docs_metadata'})
async def search(**kwargs):
    # async search logic
    return {
        'docs': [],
        'docs_metadata': {}
    }
@observation(type='generation', name="generate")
async def generate(**kwargs):
    return {
        'agent_response':'Moroccan food is delicious! I love it! Some of my favorite dishes are couscous, tagine, and harira.'
    }

@observation(type='span', name="Retriever")
async def run_retriever(**kwargs):
    result = await search(**kwargs)
    await generate(**result)

# Running the async function
async def main():
    await run_retriever(search_query='Moroccan food')

# Run the main function in asyncio event loop
asyncio.run(main())
asyncio.run(main())
