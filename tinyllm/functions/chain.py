from typing import List, Union, Dict, Type

from py2neo import NodeMatcher

from tinyllm import APP
from tinyllm.functions.function import Function
from tinyllm.state import States
from tinyllm.functions.validator import Validator

matcher = NodeMatcher(APP.graph_db)

class ChainValidator(Validator):
    children: List[Union[Function, Type[Function]]]


class Chain(Function):

    def __init__(self,
                 children,
                 **kwargs):
        super().__init__(output_validator=Validator,
                         **kwargs)
        self.children = children if children else []

    async def __call__(self, **kwargs):
        try:
            self.transition(States.INPUT_VALIDATION)
            kwargs = await self.validate_input(**kwargs)
            self.transition(States.RUNNING)
            output = None
            for child in self.children:
                output = await child(**kwargs)
                kwargs = output
            self.output = output
            self.transition(States.OUTPUT_VALIDATION)
            output = await self.validate_output(**output)
            self.transition(States.COMPLETE)
            return output
        except Exception as e:
            await self.handle_exception(e)

    @property
    def graph_state(self):
        """Returns the state of the current function and all its children."""
        graph_state = {self.name: self.state}
        for child in self.children:
            graph_state.update(child.graph_state)
        return graph_state

