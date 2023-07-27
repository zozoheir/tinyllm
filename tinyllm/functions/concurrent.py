import asyncio
from typing import List, Type, Union, Any

from tinyllm.state import States
from tinyllm.functions.function import Function, Validator


class ConcurrentValidator(Validator):
    children: List[Union[Function, Type[Function]]]


class ConcurrentInputValidator(Validator):
    inputs: List[Any]


class ConcurrentOutputValidator(Validator):
    outputs: List[Any]


class Concurrent(Function):
    def __init__(self,
                 children: List['Function'] = None,
                 **kwargs):
        m = ConcurrentValidator(children=children,
                              input_validator=ConcurrentInputValidator,
                              output_validator=ConcurrentOutputValidator,
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
        self.transition(States.OUTPUT_VALIDATION)
        output = await self.validate_output(output=output)
        self.transition(States.COMPLETE)
        return output

    def _handle_inputs_distribution(self, **kwargs):
        """
        if inputs are not provided, distribute the kwargs to all children
        """
        if 'inputs' not in kwargs: kwargs['inputs'] = [kwargs for _ in range(len(self.children))]
        return kwargs

    @property
    def graph_state(self):
        """Returns the state of the current function and all its children."""
        graph_state = {self.name: self.state}
        for child in self.children:
            graph_state.update(child.graph_state)
        return graph_state
