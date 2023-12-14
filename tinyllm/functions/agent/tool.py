import datetime as dt
from typing import Dict, Callable

from langfuse.model import CreateSpan, UpdateSpan

from tinyllm.function import Function
from tinyllm.functions.util.helpers import get_openai_message
from tinyllm.util.tracing.span import langfuse_span
from tinyllm.validator import Validator


class ToolInitValidator(Validator):
    description: str
    parameters: dict
    python_lambda: Callable


class ToolInputValidator(Validator):
    arguments: Dict


class Tool(Function):
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
        self.description = description
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

    @langfuse_span(input_key='arguments')
    async def run(self, **kwargs):
        tool_response = self.python_lambda(**kwargs['arguments'])
        return {'response': get_openai_message(role='tool', content=tool_response, name=self.name)}
