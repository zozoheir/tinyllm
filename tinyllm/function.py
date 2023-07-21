import uuid
import logging
from abc import abstractmethod
from datetime import datetime
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from tinyllm.exceptions import InvalidStateTransition
from tinyllm.types import Functions, States, ALLOWED_TRANSITIONS


class Validator(BaseModel):
    def __init__(self, **data: Any):
        if not data:
            raise ValidationError("At least one argument is required")
        super().__init__(**data)

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True


class FunctionValidator(Validator):
    name: str
    input_validator: Optional[Type[Validator]] = None
    output_validator: Optional[Type[Validator]] = None
    run_function: Optional[Callable] = None
    operator_type: Optional[Functions] = Functions.OPERATOR
    parent_id: Optional[str] = None
    log_level: Optional[int] = logging.INFO


class Function:

    def __init__(self,
                 name: str,
                 input_validator=Validator,
                 output_validator=Validator,
                 run_function: Callable = None,
                 function_type=Functions.OPERATOR,
                 parent_id=None,
                 log_level=logging.INFO):
        w = FunctionValidator(name=name,
                              input_validator=input_validator,
                              output_validator=output_validator,
                              run_function=run_function,
                              function_type=function_type,
                              parent_id=parent_id,
                              log_level=log_level)
        self.id = str(uuid.uuid4())
        self.name = name
        self.input_validator = input_validator
        self.output_validator = output_validator
        self.run = run_function if run_function is not None else self.get_output
        self.operator_type = type
        self.parent_id = parent_id
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
        self.logger = logging.getLogger(__name__)
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
        return f"[{self.parent_id}]->{self.name}[{self.id}]"

    def transition(self, new_state: States):
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(self, f"Invalid state transition from {self.state} to {new_state}")
        self.state = new_state
        self.log(f"transition to: {new_state}")

    def log(self, message, level='info'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{self.tag}-{timestamp} - {self.__class__.__name__}: {message}"
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
