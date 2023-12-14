import datetime as dt

from langfuse.model import UpdateGeneration, Usage, CreateGeneration, CreateSpan, UpdateSpan

from tinyllm.functions.util.helpers import count_tokens
from tinyllm.state import States


def langfuse_generation_stream(name=None, input_key=None, output_key=None):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get class object and parent observation
            self = args[0]
            # Get observation to log
            if kwargs.get('parent_observation', None):
                observation = kwargs['parent_observation']
            else:
                observation = self.trace
            # Generation name
            generation_name = name if name else self.__class__.__name__ + ': ' + self.name

            # Extract input using input_key
            input_value = kwargs.get(input_key, None) if input_key else kwargs.get('messages', None)

            generation = observation.generation(CreateGeneration(
                name=generation_name,
                startTime=dt.datetime.utcnow(),
                prompt=input_value,
            ))

            async_gen = func(*args, **kwargs)

            completion = ""
            function_call = {"name": None, "arguments": ""}

            async for value in async_gen:
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

            # Update generation info
            final_output = completion if output_key is None else value.get(output_key, completion)
            generation.update(UpdateGeneration(
                completion=final_output,
                endTime=dt.datetime.utcnow(),
                usage=Usage(promptTokens=count_tokens(input_value), completionTokens=count_tokens(completion)),
            ))

        return wrapper

    return decorator
