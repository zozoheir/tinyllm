import datetime as dt

from langfuse.model import UpdateGeneration, Usage, CreateGeneration

from tinyllm.util.helpers import count_tokens
from tinyllm.state import States


def langfuse_generation(func):
    async def wrapper(*args, **kwargs):
        self = args[0]

        if kwargs.get('parent_observation', None):
            observation = kwargs['parent_observation']
        else:
            observation = self.trace

        generation = observation.generation(CreateGeneration(
            name=self.name,
            prompt=kwargs['messages'],
            startTime=dt.datetime.utcnow(),
        ))

        result = await func(*args, **kwargs)
        kwargs['self'] = self
        # Evaluate
        if self.evaluators:
            self.transition(States.EVALUATING)
            for evaluator in self.evaluators:
                kwargs['self'] = self
                await evaluator(output=result, function=self, observation=generation)

        response_message = result['choices'][0]['message']
        generation.update(UpdateGeneration(
            endTime=dt.datetime.utcnow(),
            completion=response_message,
            metadata=result,
            usage=Usage(promptTokens=count_tokens(kwargs['messages']), completionTokens=count_tokens(response_message)),
        ))
        return result

    return wrapper



def langfuse_generation_stream(func):
    async def wrapper(*args, **kwargs):
        self = args[0]

        if kwargs.get('parent_observation', None):
            observation = kwargs['parent_observation']
        else:
            observation = self.trace

        # Set up initial generation tracking
        generation = observation.generation(CreateGeneration(
            name=self.name,
            startTime=dt.datetime.utcnow(),
            prompt=kwargs['messages'],
        ))

        # Call the original async generator function
        async_gen = func(*args, **kwargs)

        # Track completion and function call information
        completion = ""
        function_call = {
            "name": None,
            "arguments": ""
        }

        async for value in async_gen:
            # Process each value yielded by the generator
            if value['type'] == "completion":
                completion += value['completion']
            elif value['type'] == "tool":
                if function_call['name'] is None:
                    function_call['name'] = value['completion']['name']
                function_call['arguments'] += value['completion']['arguments']

            yield value

        if self.evaluators:
            self.transition(States.EVALUATING)
            for evaluator in self.evaluators:
                kwargs['self'] = self
                await evaluator(output=value, function=self, observation=generation)

        # Update generation info after generator is done
        generation.update(UpdateGeneration(
            completion=value,
            endTime=dt.datetime.utcnow(),
            usage=Usage(promptTokens=count_tokens(kwargs['messages']), completionTokens=count_tokens(completion)),
        ))

    return wrapper
