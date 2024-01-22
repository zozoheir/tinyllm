import datetime as dt
import inspect
import traceback
from functools import wraps

import langfuse
import numpy as np

from tinyllm import langfuse_client
from tinyllm.util.helpers import count_tokens

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


## I want you to implement an ObservationWrapper class that implements all of the above functions as class methods

class ObservationUtil:

    @classmethod
    def handle_exception(cls, obs, e):
        if 'end' in dir(obs):
            obs.end(level='ERROR', status_message=str(traceback.format_exception(e)))
        elif 'update' in dir(obs):
            obs.update(level='ERROR', status_message=str(traceback.format_exception(e)))
        raise e

    @classmethod
    def prepare_observation_input(cls, input_mapping, kwargs):
        if not input_mapping:
            # stringify values  
            kwargs = cls.keep_accepted_types(kwargs)
            return {'input': kwargs}

        return {langfuse_kwarg: kwargs[function_kwarg] for langfuse_kwarg, function_kwarg in input_mapping.items()}

    @classmethod
    def keep_accepted_types(self, d):
        acceptable_types = (str, dict, list, tuple, int, float, np.ndarray)

        def is_acceptable(v):
            if isinstance(v, acceptable_types):
                if isinstance(v, list):
                    return all(isinstance(item, acceptable_types) for item in v)
                return True
            return False

        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items() if is_acceptable(v)}
            elif isinstance(value, list):
                return [clean(item) for item in value if is_acceptable(item)]
            else:
                return value

        return clean(d)

    @classmethod
    def end_observation(cls, obs, function_input, function_output, output_mapping, observation_type, function_kwargs):
        if type(obs) == langfuse.client.StatefulTraceClient:
            return

        mapped_output = {}
        function_output = cls.keep_accepted_types(function_output)
        if not output_mapping:
            mapped_output = {'output': function_output}
        else:
            for langfuse_kwarg, function_kwarg in output_mapping.items():
                mapped_output[langfuse_kwarg] = function_output.get(function_kwarg, None)

        if observation_type == 'generation':
            prompt_tokens = count_tokens(function_input)
            completion_tokens = count_tokens(function_output)
            total_tokens = prompt_tokens + completion_tokens

            obs.end(
                end_time=dt.datetime.now(),
                model_parameters={k: v for k, v in function_kwargs.items() if
                                  k in model_parameters},
                usage={
                    'promptTokens': prompt_tokens,
                    'completionTokens': completion_tokens,
                    'totalTokens': total_tokens,
                },
                **mapped_output)
        elif observation_type == 'span':
            obs.end(**mapped_output)

    @classmethod
    async def perform_evaluations(cls, observation, result, evaluators):
        if evaluators:
            for evaluator in evaluators:
                result['observation'] = observation
                await evaluator(**result)
                result.pop('observation')

    @classmethod
    def conditional_args(cls, observation_type, input_mapping=None, output_mapping=None):
        if observation_type == 'generation':
            if input_mapping is None:
                input_mapping = {'input': 'messages'}
            if output_mapping is None:
                output_mapping = {'output': 'message'}
        return input_mapping, output_mapping

    @classmethod
    def get_obs_name(cls, *args, func):
        name = None
        # Decorated method
        if len(args) > 0:
            if hasattr(args[0], 'name'):
                name = args[0].name + ('.' + func.__name__ if func.__name__ not in ['wrapper', '__call__'] else '')
            else:
                name = args[0].__class__.__name__ + '.' + func.__name__

        # Decorated function
        else:
            if hasattr(func, '__qualname__'):
                if len(func.__qualname__.split('.')) > 1 and '<locals>' not in func.__qualname__.split('.'):
                    name = '.'.join(func.__qualname__.split('.')[-2::])
                else:
                    name = func.__name__
        return name

    @classmethod
    def get_current_obs(cls,
                        *args,
                        parent_observation,
                        observation_type,
                        name,
                        observation_input):

        if parent_observation is None:
            # This is the root function, create a new trace
            optional_args = {}
            for arg in ['user_id', 'session_id']:
                if len(args) > 0:
                    if getattr(args[0], arg, None):
                        optional_args[arg] = getattr(args[0], arg)

            observation = langfuse_client.trace(name=name,
                                                **optional_args,
                                                **observation_input)
            # Pass the trace to the Function
            if len(args) > 0:
                args[0].observation = observation
            observation_method = getattr(observation, observation_type)
            observation = observation_method(name=name, **observation_input)
        else:
            # Create child observations based on the type
            observation_method = getattr(parent_observation, observation_type)
            observation = observation_method(name=name, **observation_input)
            # Pass the parent trace to this function
            args[0].observation = parent_observation

        # Pass the generation
        if len(args) > 0:
            if hasattr(args[0], 'generation'):
                if observation_type == 'generation':
                    args[0].generation = observation

        return observation
