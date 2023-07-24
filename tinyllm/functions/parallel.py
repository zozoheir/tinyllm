import asyncio
from typing import List, Type, Union, Any

from tinyllm.types import States
from tinyllm.functions.function import Function, Validator


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

        super().__init__(**kwargs)
        self.children = children if children else []

    async def __call__(self, **kwargs):
        kwargs = self._handle_inputs_distribution(**kwargs)
        self.transition(States.INPUT_VALIDATION)
        kwargs = await self.validate_input(**kwargs)
        self.transition(States.RUNNING)
        tasks = [child.__call__(**kwargs['inputs'][i]) for i, child in enumerate(self.children)]
        output = await asyncio.gather(*tasks)
        output = await self.validate_output(ouputs=output)
        self.transition(States.COMPLETE)
        return output

    def _handle_inputs_distribution(self, **kwargs):
        # Case where we want the same input for all children functions
        # If inputs list not provided, we create it
        if 'inputs' not in kwargs: kwargs['inputs'] = [kwargs for _ in range(len(self.children))]
        return kwargs
