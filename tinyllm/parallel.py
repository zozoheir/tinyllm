import asyncio
from typing import List, Type, Union, Any

from tinyllm.exceptions import InvalidOutput, InvalidInput
from tinyllm.types import Chains, States
from tinyllm.function import Function, Validator


class ParallelValidator(Validator):
    children: List[Union[Function, Type[Function]]]


class ParallelInputValidator(Validator):
    inputs: List[Any]


class ParallelOutputValidator(Validator):
    outputs: List[Any]


class Parallel(Function):
    def __init__(self,
                 children: List['Function'] = None,
                 **kwargs):
        m = ParallelValidator(children=children,
                              input_validator=ParallelInputValidator,
                              output_validator=ParallelOutputValidator,
                              **kwargs)

        super().__init__(function_type=Chains.PARALLEL,
                         **kwargs)
        self.children = children if children else []

    async def __call__(self, **kwargs):
        self.transition(States.INPUT_VALIDATION)
        if not await self.validate_input(**kwargs):
            raise InvalidInput(self, "Invalid parallel chain input")
        self.transition(States.RUNNING)
        tasks = [child.__call__(**kwargs['inputs'][i]) for i, child in enumerate(self.children)]
        output = await asyncio.gather(*tasks)
        if not await self.validate_output(ouputs=output):
            raise InvalidOutput(self, "Invalid parallel chain output")
        self.transition(States.COMPLETE)
        return output
