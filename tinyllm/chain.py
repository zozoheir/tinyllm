from typing import List, Union, Dict, Type, Any

from enforce_typing import enforce_types

from tinyllm.function import Function
from tinyllm.types import States, Functions
from tinyllm.validator import Validator


class ChainValidator(Validator):
    children: List[Union[Function, Type[Function]]]

class ChainInputValidator(Validator):
    inputs: Dict


class Chain(Function):

    @enforce_types
    def __init__(self,
                 children,
                 **kwargs):
        super().__init__(type=Functions.CHAIN,
                         input_validator=ChainInputValidator,
                         output_validator=Validator,
                         **kwargs)
        self.children = children if children else []

    @enforce_types
    async def __call__(self, **kwargs):
        self.transition(States.INPUT_VALIDATION)
        if not await self.validate_input(**kwargs):
            raise InvalidInput(self, "Invalid sequential chain input")
        self.transition(States.RUNNING)
        output = None
        for child in self.children:
            output = await child(**kwargs)
            kwargs = output
        if not await self.validate_output(**output):
            raise InvalidOutput(self, "Invalid sequential chain output")
        self.transition(States.COMPLETE)
        return output
