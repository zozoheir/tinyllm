import datetime as dt
import traceback
from langfuse.model import UpdateGeneration, Usage, CreateGeneration

from tinyllm.tracing.span import current_parent_observation
from tinyllm.util.helpers import count_tokens
from tinyllm.state import States


async def process_evaluation(self, result, generation):
    if self.evaluators:
        self.transition(States.EVALUATING)
        for evaluator in self.evaluators:
            await evaluator(output=result, function=self, observation=generation)


async def update_generation(generation, completion, start_time, messages):
    generation.update(UpdateGeneration(
        endTime=dt.datetime.utcnow(),
        completion=completion,
        metadata=completion,
        usage=Usage(promptTokens=count_tokens(messages), completionTokens=count_tokens(completion)),
    ))
    start_time = start_time or dt.datetime.utcnow()


def create_initial_generation(self, kwargs):
    # Get or create a new parent_observation from the context
    parent_observation = current_parent_observation.get()
    if not parent_observation:
        parent_observation = self.parent_observation

    return parent_observation.generation(CreateGeneration(
        name=self.name,
        startTime=dt.datetime.utcnow(),
        prompt=kwargs['messages'],
    ))


def langfuse_generation(func):
    async def wrapper(*args, **kwargs):
        self = args[0]

        # Set the current context's parent_observation
        token = current_parent_observation.set(self.parent_observation)

        generation = create_initial_generation(self, kwargs)

        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            result = traceback.format_exception(e)

        await process_evaluation(self, result, generation)

        response_message = result['response']['choices'][0]['message']
        await update_generation(generation, response_message, None, kwargs['messages'])

        # Reset the parent_observation to the previous state after function execution
        current_parent_observation.reset(token)

        return result

    return wrapper


def langfuse_generation_stream(func):
    async def wrapper(*args, **kwargs):
        self = args[0]

        token = current_parent_observation.set(self.parent_observation)

        generation = create_initial_generation(self, kwargs)

        async_gen = func(*args, **kwargs)
        value = None

        async for val in async_gen:
            value = val
            yield value

        await process_evaluation(self, value, generation)
        await update_generation(generation, value['completion'], None, kwargs['messages'])

        current_parent_observation.reset(token)

    return wrapper
