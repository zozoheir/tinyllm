import inspect
import traceback
from functools import partial
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




class Tool(Function):

    @classmethod
    def from_pydantic_model(cls, model):
        dictionary = convert_to_openai_tool(model)
        return cls(
            name=dictionary['function']['name'],
            description=dictionary['function']['description'],
            python_lambda=lambda **kwargs: model(**kwargs).call_tool(**kwargs),
            parameters=dictionary['function']['parameters']
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

    @observation(observation_type='span')
    async def run(self, **kwargs):
        try:
            if inspect.iscoroutinefunction(self.python_lambda):
                tool_output = await self.python_lambda(**kwargs)
            else:
                tool_output = self.python_lambda(**kwargs)
        except:
            tool_output = f"""
<SYSTEM ERROR>
The tool returned the following error:
{traceback.format_exc()}
<SYSTEM ERROR>
"""

        return {'response': get_openai_message(role='tool', content=tool_output, name=self.name)}
