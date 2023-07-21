import asyncio
from typing import List, Type, Union, Any

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

        super().__init__(type=Chains.PARALLEL,
                         **kwargs)
        self.children = children if children else []

    async def __call__(self, **kwargs):
        self.transition(States.INPUT_VALIDATION)
        await self.validate_input(**kwargs)
        self.transition(States.RUNNING)
        tasks = [child.__call__(**kwargs['inputs'][i]) for i, child in enumerate(self.children)]
        output = await asyncio.gather(*tasks)
        await self.validate_output(ouputs=output)
        self.transition(States.COMPLETE)
        return output
