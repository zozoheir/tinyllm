from typing import Union, List, Any

from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator
from tinyllm.types import States


class LLMCallInitValidator(Validator):
    model_object: Any
    model_input: str


class LLMCallInputValidator(Validator):
    choices: Union[List[str], List[int]]


class LLMCallOutputValidator(Validator):
    response: Any
    tokens: Any
    cost: Any


class LLMCall(Function):
    def __init__(self,
                 model_object,
                 model_input,
                 **kwargs):
        val = LLMCallInitValidator(model_object, model_input)
        super().__init__(input_validator=LLMCallInputValidator,
                         output_validator=LLMCallOutputValidator,
                         **kwargs)

    async def __call__(self,
                       **kwargs):
        try:
            self.transition(States.INPUT_VALIDATION)
            kwargs = await self.validate_input(**kwargs)
            self.transition(States.RUNNING)
            output = await self.run(**kwargs)
            outputs = await self.validate_output(**output)
            self.transition(States.COMPLETE)
            return outputs
        except Exception as e:
            self.transition(States.FAILED)
