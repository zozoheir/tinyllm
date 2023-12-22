
from contextvars import ContextVar
from functools import wraps

from tinyllm.function import Function
from tinyllm.tracing.helpers import *

current_observation_context = ContextVar('current_observation_context', default=None)


class ObservationDecoratorFactory:

    def get_streaming_decorator(self,
                                observation_type,
                                input_mapping=None, output_mapping=None,
                                evaluators=None):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **function_input):
                parent_observation = current_observation_context.get()

                # Determine the name of the function
                name = get_obs_name(*args, func=func)

                # Prepare the input for the observation
                observation_input = prepare_observation_input(input_mapping, function_input)

                # Get the current observation
                observation = get_current_obs(parent_observation=parent_observation,
                                              observation_type=observation_type,
                                              name=name,
                                              function_input=observation_input)
                # Pass the observation to the class (so it can evaluate it)
                function_input['observation'] = observation
                # Set the current observation in the context for child functions to access
                token = current_observation_context.set(observation)
                try:
                    async for result in  func(*args, **function_input):
                        yield result
                    function_input.pop('observation')
                    await perform_evaluations(observation, result, evaluators)
                except Exception as e:
                    handle_exception(observation, e)
                finally:
                    current_observation_context.reset(token)
                    end_observation(observation, observation_input, result, output_mapping, observation_type, function_input)

            return wrapper

        return decorator


    @staticmethod
    def get_observation_decorator(observation_type, input_mapping=None, output_mapping=None,
                                  evaluators=None, stream=False):

        if stream:
            return ObservationDecoratorFactory().get_streaming_decorator(observation_type,
                                                                         input_mapping=input_mapping,
                                                                         output_mapping=output_mapping,
                                                                         evaluators=evaluators)

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **function_input):
                parent_observation = current_observation_context.get()

                # Determine the name of the function
                name = get_obs_name(*args, func=func)

                # Prepare the input for the observation
                observation_input = prepare_observation_input(input_mapping, function_input)

                # Get the current observation
                observation = get_current_obs(parent_observation,
                                              observation_type,
                                              name,
                                              observation_input)

                # Set the current observation in the context for child functions to access
                token = current_observation_context.set(observation)
                if len(args) > 0:
                    if isinstance(args[0], Function):
                        args[0].observation = observation
                try:
                    result = await func(*args, **function_input)
                    await perform_evaluations(observation, result, evaluators)
                    return result
                except Exception as e:
                    handle_exception(observation, e)
                finally:
                    current_observation_context.reset(token)
                    end_observation(observation, observation_input, result, output_mapping, observation_type, function_input)

            return wrapper

        return decorator


# Decorator utility function for easy use
def observation(observation_type='span', input_mapping=None, output_mapping=None, evaluators=None, stream=False):
    input_mapping, output_mapping = conditional_args(observation_type, input_mapping, output_mapping)
    return ObservationDecoratorFactory.get_observation_decorator(observation_type=observation_type,
                                                                 input_mapping=input_mapping,
                                                                 output_mapping=output_mapping,
                                                                 evaluators=evaluators,
                                                                 stream=stream)
