import uuid
from typing import Any

from tinyllm import Operators, States, ALLOWED_TRANSITIONS
from tinyllm.exceptions import InvalidOutput, InvalidInput, InvalidStateTransition


class Operator:

    def __init__(self, name: str, type: Operators = Operators.OPERATOR, parent_id=None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.type = type
        self.parent_id = parent_id
        self.children = []
        self.state = States.INIT
        self.io_history = []

    async def __call__(self, **kwargs):
        try:
            self.state = States.INPUT_VALIDATION
            self.input = kwargs
            if await self.validate_input(**kwargs):
                self.transition(States.RUNNING)
                output = await self.get_output(**kwargs)
                if await self.validate_output(**output):
                    self.transition(States.COMPLETE)
                    self.output = output
                else:
                    self.transition(States.FAILED)
            else:
                self.transition(States.FAILED)
        except Exception as e:
            self.transition(States.FAILED)
            raise e
        return output

    @property
    def tag(self):
        return f"[{self.parent_id}]->{self.name}[{self.id}]"

    def transition(self, new_state):
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(self, f"Invalid state transition from {self.state} to {new_state}")
        self.state = new_state

    async def validate_init(self, **kwargs) -> InvalidInput:
        return bool(kwargs)

    async def validate_input(self, **kwargs) -> InvalidInput:
        return bool(kwargs)

    async def validate_output(self, **kwargs: Any) -> InvalidOutput:
        return bool(kwargs)

    async def get_output(self, **kwargs):
        return bool(kwargs)

