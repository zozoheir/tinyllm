from typing import List, Union, Dict, Type

from enforce_typing import enforce_types

from tinyllm.functions.function import Function
from tinyllm.types import States
from tinyllm.functions.validator import Validator


class ChainValidator(Validator):
    children: List[Union[Function, Type[Function]]]


class Chain(Function):

    @enforce_types
    def __init__(self,
                 children,
                 **kwargs):
        super().__init__(output_validator=Validator,
                         **kwargs)
        self.children = children if children else []

    @enforce_types
    async def __call__(self, **kwargs):
        self.transition(States.INPUT_VALIDATION)
        kwargs = await self.validate_input(**kwargs)
        self.transition(States.RUNNING)
        output = None
        for child in self.children:
            output = await child(**kwargs)
            kwargs = output
        is_valid = await self.validate_output(**output)
        self.transition(States.COMPLETE)
        return output
