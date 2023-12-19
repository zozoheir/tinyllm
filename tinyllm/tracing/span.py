from contextvars import ContextVar

import datetime as dt
import traceback
from langfuse.model import CreateSpan, UpdateSpan
from tinyllm import tinyllm_config
from tinyllm.state import States


current_parent_observation: ContextVar = ContextVar('current_parent_observation', default=None)


def langfuse_span(name=None, input_key=None, output_key=None, visual_output_lambda=lambda x: x, evaluators=None):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            self = args[0]

            # Get or create a new parent_observation from the context
            parent_observation = current_parent_observation.get()
            if not parent_observation:
                parent_observation = self.parent_observation

            span_name = self.name if name is None else name
            input_value = kwargs if input_key is None else kwargs.get(input_key, None)

            # Create a span
            span = CreateSpan(
                name=span_name,
                startTime=dt.datetime.utcnow(),
                input=input_value
            )
            self.parent_observation = parent_observation.span(span)

            # Set the current context's parent_observation
            token = current_parent_observation.set(self.parent_observation)

            exception_msg = None
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                exception_msg = str(traceback.format_exception(e))
                self.parent_observation.update(
                    UpdateSpan(
                        output=exception_msg,
                        endTime=dt.datetime.utcnow()
                    )
                )
                if tinyllm_config['OPS']['DEBUG']:
                    raise e
            finally:
                # Reset the parent_observation to the previous state after function execution
                current_parent_observation.reset(token)

            if exception_msg is None:
                output_value = result if output_key is None else result.get(output_key, None)
                visual_output = visual_output_lambda(output_value)
                self.parent_observation.update(
                    UpdateSpan(
                        output=visual_output,
                        metadata=output_value,
                        endTime=dt.datetime.utcnow()
                    )
                )

            if evaluators is None:
                evaluators_to_use = self.evaluators
            else:
                evaluators_to_use = evaluators

            if evaluators_to_use:
                self.transition(States.EVALUATING)
                for evaluator in evaluators_to_use:
                    kwargs['self'] = self
                    await evaluator(output=output_value, function=self, observation=self.parent_observation)

            return result if exception_msg is None else exception_msg

        return wrapper
    return decorator
