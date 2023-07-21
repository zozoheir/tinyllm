from typing import List, Union, Dict

from enforce_typing import enforce_types

from tinyllm.exceptions import InvalidInput, InvalidOutput
from tinyllm.function import Function
from tinyllm.types import States


class Chain(Function):

    @enforce_types
    def __init__(self, children: List[Function] = None, **kwargs):
        super().__init__(**kwargs)
        self.children = children if children else []

    @enforce_types
    async def __call__(self, input: Dict):
        self.transition(States.INPUT_VALIDATION)
        if not await self.validate_input(**input):
            raise InvalidInput(self, "Invalid sequential chain input")
        self.transition(States.RUNNING)
        output = None
        for child in self.children:
            output = await child(**input)
            kwargs = output
        if not await self.validate_output(**output):
            raise InvalidOutput(self, "Invalid sequential chain output")
        self.transition(States.COMPLETE)
        return output

    @enforce_types
    async def validate_input(self, input: Union[dict, List] = None) -> bool:
        return True
