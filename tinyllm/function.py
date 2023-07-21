import uuid
import logging
from abc import abstractmethod
from typing import Any, Callable, Optional, Type

from tinyllm.config import APP_CONFIG
from tinyllm.exceptions import InvalidStateTransition
from tinyllm.types import States, ALLOWED_TRANSITIONS
from tinyllm.validator import Validator

class FunctionConfigValidator(Validator):
    type: str
    name: str
    input_validator: Optional[Type[Validator]]
    output_validator: Optional[Type[Validator]]
    run_function: Optional[Callable]
    parent_id: Optional[str]


class Function:

    def __init__(self,
                 type,
                 name,
                 input_validator=Validator,
                 output_validator=Validator,
                 run_function=None,
                 parent_id=None):
        w = FunctionConfigValidator(
            type=type,
            name=name,
            input_validator=input_validator,
            output_validator=output_validator,
            run_function=run_function,
            parent_id=parent_id)
        self.id = str(uuid.uuid4())
        self.logger = APP_CONFIG.logging['default']
        self.name = name
        self.input_validator = input_validator
        self.output_validator = output_validator
        self.run = run_function if run_function is not None else self.get_output
        self.type = type
        self.parent_id = parent_id
        self.state = None
        self.transition(States.INIT)


    async def __call__(self, **kwargs):
        try:
            self.transition(States.INPUT_VALIDATION)
            self.input = kwargs
            if await self.validate_input(**kwargs):
                self.transition(States.RUNNING)
                output = await self.run(**kwargs)
                if await self.validate_output(**output):
                    self.transition(States.COMPLETE)
                    self.output = output
                else:
                    self.transition(States.FAILED)
            else:
                self.transition(States.FAILED)
        except Exception as e:
            self.transition(States.FAILED)
            self.log("Exception occurred", level='error')
            raise e
        return output

    @property
    def tag(self):
        return f""

    def transition(self, new_state: States):
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(self, f"Invalid state transition from {self.state} to {new_state}")
        self.state = new_state
        self.log(f"transition to: {new_state}")

    def log(self, message, level='info'):
        if self.logger is None:
            return
        log_message = f"{self.name}[id:{self.id}]: {message}"
        if level == 'error':
            self.logger.error(log_message)
        else:
            self.logger.info(log_message)

    async def validate_input(self, **kwargs: int) -> bool:
        self.input_validator(**kwargs)
        dir(Validator)
        return True

    async def validate_output(self, **kwargs) -> bool:
        self.output_validator(**kwargs)
        return True

    @abstractmethod
    async def get_output(self, **kwargs) -> Any:
        pass
