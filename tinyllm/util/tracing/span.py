import datetime as dt

from langfuse.model import UpdateGeneration, Usage, CreateGeneration, CreateSpan, UpdateSpan

from tinyllm.functions.util.helpers import count_tokens
from tinyllm.state import States



def langfuse_span(name=None, input_key=None, output_key=None):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            self = args[0]

            span_name = name if name else self.__class__.__name__+': '+self.name
            if kwargs.get('parent_observation', None):
                observation = kwargs['parent_observation']
            else:
                observation = self.trace

            # Extract input using input_key
            input_value = kwargs.get(input_key, None)

            # Create a span
            span = observation.span(
                CreateSpan(
                    name=span_name,
                    startTime=dt.datetime.utcnow(),
                    input=input_value
                )
            )
            self.parent_observation = span

            # Call the original function
            result = await func(*args, **kwargs)

            # Extract output using output_key
            output_value = result if output_key is None else result.get(output_key, None)

            # Update the span with endTime and output
            span.update(
                UpdateSpan(
                    endTime=dt.datetime.utcnow(),
                    output=output_value
                )
            )
            # Evaluate
            if self.evaluators:
                self.transition(States.EVALUATING)
                for evaluator in self.evaluators:
                    kwargs['self'] = self
                    await evaluator(output=output_value, function=self, observation=span)

            return result

        return wrapper

    return decorator
