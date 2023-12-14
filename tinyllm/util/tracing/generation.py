#https://langfuse.com/docs/sdk/python

import json
import datetime as dt

from langfuse.model import UpdateGeneration, Usage, CreateGeneration, CreateSpan, UpdateSpan

from tinyllm import langfuse_client
from tinyllm.functions.util.helpers import count_tokens
from tinyllm.state import States


def langfuse_generation(name=None, prompt_key='messages', completion_key=None):
    def decorator(func):
        async def wrapper(*args, **kwargs):

            # Get class object and parent observation
            self = args[0]
            # Get observation to log
            if kwargs.get('parent_observation', None):
                observation = kwargs['parent_observation']
            else:
                observation = self.trace

            langfuse_generation_args = {
                'name': name if name else self.__class__.__name__ + ': ' + self.name,
                'prompt': kwargs[prompt_key],
                'completion': None,
                'usage': None,
                'startTime': dt.datetime.utcnow(),
                'endTime': None,
                'model': kwargs['model'],
                'modelParameters': {k:v for k,v in kwargs.items() if k in ['model','max_tokens','temperature']},
                'metadata': None,
                'level': None,
                'version': None,
            }

            prompt = kwargs[prompt_key]
            for msg in kwargs[prompt_key]:
                if 'tool_calls' in msg:
                    prompt = json.dumps(prompt)
                    break

            generation = observation.generation(CreateGeneration(
                name=name if name else self.__class__.__name__ + ': ' + self.name,
                prompt=prompt,
                startTime=dt.datetime.utcnow(),
                model=kwargs['model'],
                modelParameters={k:v for k,v in kwargs.items() if k in ['model','max_tokens','temperature']},
            ))
            dir(generation)

            result = await func(*args, **kwargs)

            # Completion
            langfuse_generation_args['completion'] = result['choices'][0]['message']
            if langfuse_generation_args['completion']['content'] is None:
                langfuse_generation_args['completion']['content'] = ''
            # Update the generation info
            generation.update(UpdateGeneration(
                endTime=dt.datetime.utcnow(),
                completion=langfuse_generation_args['completion'],
                usage=Usage(promptTokens=count_tokens(langfuse_generation_args['prompt']), completionTokens=count_tokens(langfuse_generation_args['completion'])),
            ))
            langfuse_client.flush()

            # Evaluate
            if self.evaluators:
                self.transition(States.EVALUATING)
                for evaluator in self.evaluators:
                    kwargs['self'] = self
                    await evaluator(output=langfuse_generation_args['completion'], function=self, observation=generation)

            return result

        return wrapper

    return decorator
