from contextvars import ContextVar

from pydantic import BaseModel

from smartpy.utility.py_util import stringify_values_recursively
from tinyllm.tracing.helpers import *

current_observation_context = ContextVar('current_observation_context', default=None)


class ObservationDecoratorFactory:

    @classmethod
    def get_streaming_decorator(self,
                                observation_type,
                                input_mapping=None,
                                output_mapping=None,
                                evaluators=None):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **function_input):
                parent_observation = current_observation_context.get()

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
                if len(args) > 0:
                    args[0].observation = observation
                # Set the current observation in the context for child functions to access
                token = current_observation_context.set(observation)
                result = {}
                try:
                    async for result in func(*args, **function_input):
                        yield result
                    if type(result) != dict: result = {'result': result}
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
                      name=None,
                      input_mapping=None,
                      output_mapping=None,
                      evaluators=None):
        def decorator(func):

            @wraps(func)
            async def wrapper(*args, **function_input):
                parent_observation = current_observation_context.get()
                if name is None:
                    obs_name = ObservationUtil.get_obs_name(*args, func=func)
                else:
                    obs_name = name
                observation_input = ObservationUtil.prepare_observation_input(input_mapping, function_input)
                observation = ObservationUtil.get_current_obs(*args,
                                                              parent_observation=parent_observation,
                                                              observation_type=observation_type,
                                                              name=obs_name,
                                                              observation_input=observation_input)
                token = current_observation_context.set(observation)
                result = {}
                if len(args) > 0:
                    args[0].observation = observation
                try:
                    result = await func(*args, **function_input)
                    if type(result) != dict: result = {'result': result}
                    # convert pydantic models to dict
                    for key, value in result.items():
                        if isinstance(value, BaseModel):
                            result[key] = value.model_dump()

                    await ObservationUtil.perform_evaluations(observation, result, evaluators)
                    return result
                except Exception as e:
                    ObservationUtil.handle_exception(observation, e)
                    raise e
                finally:
                    current_observation_context.reset(token)
                    ObservationUtil.end_observation(observation, observation_input, result, output_mapping,
                                                    observation_type, function_input)
                    langfuse_client.flush()

            return wrapper

        return decorator


def observation(observation_type='span',name=None, input_mapping=None, output_mapping=None, evaluators=None, stream=False):
    input_mapping, output_mapping = ObservationUtil.conditional_args(observation_type,
                                                                     input_mapping,
                                                                     output_mapping)
    if stream:
        return ObservationDecoratorFactory.get_streaming_decorator(observation_type, input_mapping, output_mapping,
                                                                   evaluators)
    else:
        return ObservationDecoratorFactory.get_decorator(observation_type, name, input_mapping, output_mapping, evaluators)
