from contextvars import ContextVar
from tinyllm.tracing.helpers import *

current_observation_context = ContextVar('current_observation_context', default=None)

model_parameters = [
    "model",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_p"
]


class ObservationDecoratorFactory:

    @classmethod
    def get_streaming_decorator(self,
                                observation_type,
                                input_mapping=None, output_mapping=None,
                                evaluators=None):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **function_input):
                parent_observation = current_observation_context.get()

                # Determine the name of the function
                name = ObservationUtil.get_obs_name(*args, func=func)

                # Prepare the input for the observation
                observation_input = ObservationUtil.prepare_observation_input(input_mapping, function_input)

                # Get the current observation
                observation = ObservationUtil.get_current_obs(*args,
                                                              parent_observation=parent_observation,
                                                              observation_type=observation_type,
                                                              name=name,
                                                              observation_input=observation_input)
                # Pass the observation to the class (so it can evaluate it)
                function_input['observation'] = observation
                # Set the current observation in the context for child functions to access
                token = current_observation_context.set(observation)
                try:
                    async for result in func(*args, **function_input):
                        yield result
                    function_input.pop('observation')
                    await ObservationUtil.perform_evaluations(observation, result, evaluators)
                except Exception as e:
                    ObservationUtil.handle_exception(observation, e)
                finally:
                    current_observation_context.reset(token)
                    ObservationUtil.end_observation(observation, observation_input, result, output_mapping,
                                                    observation_type, function_input)

            return wrapper

        return decorator

    @classmethod
    def get_decorator(self,
                      observation_type,
                      input_mapping=None,
                      output_mapping=None,
                      evaluators=None,
                      **kwargs):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **function_input):
                parent_observation = current_observation_context.get()
                name = ObservationUtil.get_obs_name(*args, func=func)
                observation_input = ObservationUtil.prepare_observation_input(input_mapping, function_input)
                observation = ObservationUtil.get_current_obs(*args,
                                                              parent_observation=parent_observation,
                                                              observation_type=observation_type,
                                                              name=name,
                                                              observation_input=observation_input)
                token = current_observation_context.set(observation)
                if len(args) > 0:
                    if args and type(args[0]).__name__ == 'Function':
                        args[0].observation = observation
                try:
                    result = await func(*args, **function_input)
                    await ObservationUtil.perform_evaluations(observation, result, evaluators)
                    return result
                except Exception as e:
                    ObservationUtil.handle_exception(observation, e)
                finally:
                    current_observation_context.reset(token)
                    ObservationUtil.end_observation(observation, observation_input, result, output_mapping,
                                                    observation_type, function_input)

            return wrapper

        return decorator


# Decorator utility function for easy use
def observation(observation_type='span', input_mapping=None, output_mapping=None, evaluators=None, stream=False):
    input_mapping, output_mapping = ObservationUtil.conditional_args(observation_type,
                                                                     input_mapping,
                                                                     output_mapping)
    if stream:
        return ObservationDecoratorFactory.get_streaming_decorator(observation_type, input_mapping, output_mapping,evaluators)
    else:
        return ObservationDecoratorFactory.get_decorator(observation_type, input_mapping, output_mapping,evaluators)
