import inspect
from typing import Dict, Callable

from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool

from tinyllm.function import Function
from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message
from tinyllm.validator import Validator


class ToolInitValidator(Validator):
    description: str
    parameters: dict
    python_lambda: Callable


class ToolInputValidator(Validator):
    arguments: Dict


class Tool(Function):

    @classmethod
    def from_pydantic_model(cls, model):
        dictionary = convert_to_openai_tool(model)
        return cls(
            name=dictionary['name'],
            description=dictionary['description'],
            python_lambda=model.function,
            parameters=dictionary['parameters']
        )


    def __init__(self,
                 description,
                 parameters,
                 python_lambda,
                 **kwargs):
        ToolInitValidator(
            description=description,
            parameters=parameters,
            python_lambda=python_lambda,
        )
        super().__init__(
            input_validator=ToolInputValidator,
            **kwargs)
        self.description = description.strip()
        self.parameters = parameters
        self.python_lambda = python_lambda

    def as_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    @observation(observation_type='span', input_mapping={'input': 'arguments'})
    async def run(self, **kwargs):
        if inspect.iscoroutinefunction(self.python_lambda):
            tool_response = await self.python_lambda(**kwargs['arguments'])
        else:
            tool_response = self.python_lambda(**kwargs['arguments'])
        return {'response': get_openai_message(role='tool', content=tool_response, name=self.name)}
