import abc
from typing import List, Dict, Any
from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator


class InitValidator(Validator):
    prompt_template: Any


class OutputValidator(Validator):
    response: Dict[str, Any]


class LLMCall(Function, abc.ABC):

    def __init__(self,
                 prompt_template: List[Dict[str, str]],
                 **kwargs):
        val = InitValidator(prompt_template=prompt_template)
        super().__init__(
            output_validator=OutputValidator,
            **kwargs)
        self.prompt_template = prompt_template

    async def run(self, message: str):
        # Your code here
        pass

    async def process_output(self, response: Dict[str, Any]):
        # Your code here
        pass
