import datetime as dt
import traceback

from langfuse.model import CreateSpan, UpdateSpan

from tinyllm.state import States


def langfuse_span(name=None, input_key=None, output_key=None, visual_output_lambda=lambda x: x, evaluators=None):
    def decorator(func):
        async def wrapper(*args, **kwargs):

            # Extract arguments
            self = args[0]
            span_name = self.name if name is None else name
            if kwargs.get('parent_observation', None):
                observation = kwargs['parent_observation']
            else:
                observation = self.trace

            # Extract input
            input_value = kwargs if input_key is None else kwargs.get(input_key, None)

            # Create a span
            span = observation.span(
                CreateSpan(
                    name=span_name,
                    startTime=dt.datetime.utcnow(),
                    input=input_value
                )
            )
            self.parent_observation = span

            ###### Call the original function ######
            exception_msg = None
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                exception_msg = traceback.format_exception(e)


            # Extract output
            if exception_msg is None:
                output_value = result if output_key is None else result.get(output_key, None)
                visual_output = visual_output_lambda(output_value)
            else:
                output_value = exception_msg
                visual_output = exception_msg


            # Update the span
            span.update(
                UpdateSpan(
                    output=visual_output,
                    metadata=output_value,
                    endTime=dt.datetime.utcnow()
                )
            )

            # Evaluate
            if evaluators is None:
                evaluators_to_use = self.evaluators
            else:
                evaluators_to_use = evaluators

            if evaluators_to_use:
                self.transition(States.EVALUATING)
                for evaluator in evaluators_to_use:
                    kwargs['self'] = self
                    await evaluator(output=output_value, function=self, observation=span)

            return result if exception_msg is None else exception_msg

        return wrapper

    return decorator


def langfuse_span_generator(name=None, input_key=None, output_key=None, visual_output_lambda=lambda x:x):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            self = args[0]
            span_name = self.name if name is None else name
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

            # Call the original generator function
            async_gen = func(*args, **kwargs)

            async for result in async_gen:
                # Extract output using output_key
                output_value = result if output_key is None else result.get(output_key, None)
                yield result

            # After the generator is exhausted, update the span with endTime
            visual_output = visual_output_lambda(output_value)
            span.update(
                UpdateSpan(
                    output=visual_output,
                    metadata=output_value,
                    endTime=dt.datetime.utcnow()
                )
            )
            # Evaluate (if applicable)
            if self.evaluators:
                self.transition(States.EVALUATING)
                for evaluator in self.evaluators:
                    kwargs['self'] = self
                    await evaluator(output=output_value, function=self, observation=span)

        return wrapper

    return decorator
