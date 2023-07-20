import asyncio
from typing import List

from tinyllm import States, Chains
from tinyllm.exceptions import InvalidOutput, InvalidInput
from tinyllm.operator import Operator

class ParallelChain(Operator):
    def __init__(self, name: str, children: List['Operator'] = None, parent_id=None):
        super().__init__(name, Chains.PARALLEL, parent_id)
        self.children = children if children else []

    async def __call__(self, *args, **kwargs):
        self.transition(States.INPUT_VALIDATION)
        if not await self.validate_input(*args):
            raise InvalidInput(self, "Invalid parallel chain input")
        self.transition(States.RUNNING)
        output = await asyncio.gather(*(child(**args[0][i]) for i, child in enumerate(self.children)))
        if not await self.validate_output(*output):
            raise InvalidOutput(self, "Invalid parallel chain output")
        self.transition(States.COMPLETE)
        return output

    async def validate_input(self, *args, **kwargs):
        if not isinstance(args[0], list) or not all(isinstance(i, dict) for i in args[0]):
            raise InvalidInput(self, "Invalid input type. Expected list of dictionaries.")
        return True