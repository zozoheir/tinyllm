import uuid
from typing import Any, Callable, Optional, Type, Dict

from py2neo import Node
from pydantic import field_validator

from tinyllm.app import APP
from tinyllm.exceptions import InvalidStateTransition
from tinyllm.helpers import concatenate_strings
from tinyllm.state import States, ALLOWED_TRANSITIONS
from tinyllm.functions.validator import Validator
from inspect import iscoroutinefunction


class FunctionInitValidator(Validator):
    name: str
    input_validator: Optional[Type[Validator]]
    output_validator: Optional[Type[Validator]]
    run_function: Optional[Callable]
    parent_id: Optional[str]
    verbose: bool

    @field_validator('run_function')
    def validate_run_function(cls, v):
        if v is not None and not iscoroutinefunction(v):
            raise ValueError('run_function must be an async function')
        return v


class Function:

    def __init__(self,
                 name,
                 input_validator=Validator,
                 output_validator=Validator,
                 run_function=None,
                 parent_id=None,
                 verbose=True):
        w = FunctionInitValidator(
            name=name,
            input_validator=input_validator,
            output_validator=output_validator,
            run_function=run_function,
            parent_id=parent_id,
            verbose=verbose)
        self.id = str(uuid.uuid4())
        self.logger = APP.logging['default']
        self.name = name

        self.input_validator = input_validator
        self.output_validator = output_validator
        self.run_function = run_function if run_function is not None else self.run
        self.parent_id = parent_id
        self.verbose = verbose
        self.logs = ""
        self.state = None
        self.transition(States.INIT)
        self.error_message = None

    @property
    def graph_state(self):
        """Returns the state of the current function."""
        return {self.name: self.state}

    async def __call__(self, **kwargs):
        try:
            self.transition(States.INPUT_VALIDATION)
            kwargs = await self.validate_input(**kwargs)
            self.transition(States.RUNNING)
            output = await self.run_function(**kwargs)
            self.transition(States.OUTPUT_VALIDATION)
            output = await self.validate_output(**output)
            self.transition(States.PROCESSING_OUTPUT)
            output = await self.process_output(**output)
            self.transition(States.COMPLETE)
            await self.push_to_db()
            return output
        except Exception as e:
            self.error_message = str(e)
            self.transition(States.FAILED,
                            msg=str(e))
            await self.push_to_db()

    def transition(self, new_state: States,
                   msg: Optional[str] = None):
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(self, f"Invalid state transition from {self.state} to {new_state}")
        self.state = new_state
        self.log(f"transition to: {new_state}" + (f" ({msg})" if msg is not None else ""))

    def log(self, message, level='info'):
        if self.logger is None or self.verbose is False:
            return
        log_message = f"[{self.name}] {message}"
        self.logs += '\n' + log_message
        if level == 'error':
            self.logger.error(log_message)
        else:
            self.logger.info(log_message)

    async def validate_input(self, **kwargs):
        return self.input_validator(**kwargs).model_dump()

    async def validate_output(self, **kwargs):
        return self.output_validator(**kwargs).model_dump()

    async def push_to_db(self):
        attributes_dict = vars(self)
        attributes_dict = {key: str(value) for key, value in attributes_dict.items()}
        node = Node(self.__class__.__name__, **attributes_dict)
        APP.graph_db.create(node)

    async def run(self, **kwargs) -> Any:
        pass

    async def process_output(self, **kwargs):
        return kwargs
