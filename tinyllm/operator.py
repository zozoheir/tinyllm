import uuid
import logging
from abc import abstractmethod
from datetime import datetime
from typing import Any

from tinyllm.exceptions import InvalidStateTransition
from tinyllm.types import Operators, States, ALLOWED_TRANSITIONS


class Operator:

    def __init__(self, name: str, operator_type: Operators = Operators.OPERATOR, parent_id=None, log_level=logging.INFO):

        self.id = str(uuid.uuid4())
        self.name = name
        self.type = operator_type
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

    async def validate_input(self, *args, **kwargs) -> bool:
        return True

    @abstractmethod
    async def get_output(self, *args, **kwargs) -> Any:
        pass

    async def validate_output(self, *args, **kwargs: Any) -> bool:
        return True
