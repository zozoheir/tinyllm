import uuid
import asyncio
from typing import List

from tinyllm import States, Chains, Operators
from tinyllm.exceptions import InvalidInput, InvalidOutput, OperatorError
from tinyllm.operator import Operator




class Chain(Operator):
    def __init__(self, name: str, type: Chains, children: List['Operator'] = None, parent_id=None):
        super().__init__(name=name,
                         type=type,
                         parent_id=parent_id)
        self.children = children if children else []
        self.state = States.INIT


    async def __call__(self, **kwargs):
        try:
            self.transition(States.INPUT_VALIDATION)
            if not await self.validate_input(**kwargs):
                raise InvalidInput("Invalid chain input")

            self.transition(States.RUNNING)
            output = None
            if self.type == Chains.SEQUENTIAL:
                for child in self.children:
                    output = await child(**kwargs)
                    kwargs = output
            elif self.type == Chains.PARALLEL:
                output = await asyncio.gather(*(child(**kwargs) for child in self.children))

            if not await self.validate_output(**output):
                raise InvalidOutput(self, "Invalid chain output")

            self.transition(States.COMPLETE)
            return output
        except Exception as e:
            self.transition(States.FAILED)
            raise OperatorError(self, f"Chain error error: {e}")

    async def validate_input(self, **kwargs):
        return True

    async def validate_output(self, **kwargs):
        return True

