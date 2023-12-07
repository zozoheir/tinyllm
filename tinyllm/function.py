"""
Function is the building block of tinyllm. A function has 4 main components:
    1. A name: required
    2. An input validator: optional, but recommended
    3. An output validator: optional, but recommended
    4. A run function: required

- Chains and Concurrent inherit from Functions.
- When creating Functions or child classes, the above requirements apply
- Functions, Chains, and Concurrents and
"""
import datetime as dt
import uuid
from typing import Any, Optional, Type

import pydantic
import pytz
from langfuse.api import CreateDatasetRequest, CreateDatasetItemRequest
from langfuse.client import DatasetItemClient
from langfuse.model import CreateTrace

from smartpy.utility.log_util import getLogger
from smartpy.utility.py_util import get_exception_info
from tinyllm.exceptions import InvalidStateTransition
from tinyllm.llm_ops import LLMTrace, langfuse_client
from tinyllm.state import States, ALLOWED_TRANSITIONS
from tinyllm.validator import Validator
from tinyllm.util.fallback_strategy import fallback_decorator


def pretty_print(value):
    if isinstance(value, dict):
        return {key: pretty_print(val) for key, val in value.items()}
    elif isinstance(value, list):
        return [pretty_print(val) for val in value]
    else:
        return value


class FunctionInitValidator(Validator):
    name: str
    input_validator: Optional[Type[Validator]]
    output_validator: Optional[Type[Validator]]
    is_traced: bool
    debug: bool
    evaluators: Optional[list]
    dataset_name: Optional[str]
    stream: Optional[bool]


class DefaultInputValidator(Validator):
    role: Any
    content: Any


class DefaultOutputValidator(Validator):
    response: Any


class Function:

    def __init__(
            self,
            name,
            user_id=None,
            input_validator=Validator,
            output_validator=DefaultOutputValidator,
            evaluators=[],
            dataset_name=None,
            is_traced=True,
            debug=True,
            required=True,
            stream=False,
            trace=None,
            fallback_strategies={},

    ):
        w = FunctionInitValidator(
            name=name,
            input_validator=input_validator,
            output_validator=output_validator,
            is_traced=is_traced,
            debug=debug,
            stream=stream,
        )
        self.user = user_id
        self.init_timestamp = dt.datetime.now(pytz.UTC).isoformat()
        self.function_id = str(uuid.uuid4())
        self.logger = getLogger(__name__)
        self.name = name

        self.input_validator = input_validator
        self.output_validator = output_validator
        self.is_traced = is_traced
        self.required = required
        self.logs = ""
        self.state = None
        self.transition(States.INIT)
        self.error_message = None
        self.input = None
        self.output = None
        self.processed_output = None
        self.scores = []
        self.trace = None
        self.debug = debug
        if trace is None and is_traced is True:
            self.trace = langfuse_client.trace(CreateTrace(
                name=self.name,
                userId="test")
            )
            self.generation = None
        else:
            self.trace = trace

        self.cache = {}

        self.evaluators = evaluators
        self.dataset_name = dataset_name
        self.dataset = None

        if self.dataset_name is not None:
            try:
                self.dataset = langfuse_client.get_dataset(name=dataset_name)
            except pydantic.error_wrappers.ValidationError:
                self.dataset = langfuse_client.create_dataset(CreateDatasetRequest(name=dataset_name))

        self.fallback_strategies = fallback_strategies
        self.stream = stream
        self.generation = None

    @fallback_decorator
    async def __call__(self, **kwargs):
        try:
            self.input = kwargs
            self.transition(States.INPUT_VALIDATION)
            validated_input = await self.validate_input(**kwargs)
            self.transition(States.RUNNING)
            self.output = await self.run(**validated_input)
            self.transition(States.OUTPUT_VALIDATION)
            self.output = await self.validate_output(**self.output)
            self.transition(States.PROCESSING_OUTPUT)
            self.processed_output = await self.process_output(**self.output)
            final_output = {"status": "success",
                            "output": self.processed_output}
            if self.evaluators:
                self.transition(States.EVALUATING)
                await self.evaluate(generation=self.generation,
                                    output=final_output,
                                    **kwargs)

            self.transition(States.COMPLETE)
            langfuse_client.flush()
            return final_output
        except Exception as e:
            self.error_message = str(e)
            self.transition(States.FAILED, msg=str(e))
            detailed_error_msg = get_exception_info(e)
            self.log(detailed_error_msg, level="error")
            langfuse_client.flush()
            if type(e) in self.fallback_strategies:
                raise e
            else:
                return {"status": "error",
                        "message": detailed_error_msg}

    def transition(self, new_state: States, msg: Optional[str] = None):
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(
                self, f"Invalid state transition from {self.state} to {new_state}"
            )
        self.state = new_state
        log_level = "error" if new_state == States.FAILED else "info"
        self.log(
            f"transition to: {new_state}" + (f" ({msg})" if msg is not None else ""),
            level=log_level,
        )

    async def evaluate(self,
                       **kwargs):
        generation = kwargs['generation']
        output = kwargs['output']

        # Need to link item with generation + call evaluators

        # Create dataset item
        if self.dataset:
            item = langfuse_client.create_dataset_item(
                CreateDatasetItemRequest(dataset_name=self.dataset_name,
                                         input=kwargs.get('input', "None"),
                                         expected_output=kwargs.get('expected_output', "None")
                                                         ** kwargs)
            )
            item_client = DatasetItemClient(item, langfuse_client)
            item_client.link(generation, kwargs.get('run_name', "tinyllm_function"))

        # Create eval_data args
        eval_kwargs = {
            'cache': self.cache,
        }
        if self.processed_output:
            eval_kwargs['processed_output'] = self.processed_output

        if kwargs:
            eval_kwargs['kwargs'] = kwargs

        # Call evaluators and score
        for evaluator in self.evaluators:
            evaluator_response = await evaluator(generation=generation,
                                                 output=output,
                                                 **eval_kwargs)
            if evaluator_response['status'] != 'success':
                self.log(evaluator_response['message'], level="error")

    def log(self, message, level="info"):
        # Only log if debug
        if getattr(self, 'debug', None):
            log_message = f"[{self.name}] {message}"
            if getattr(self, 'trace', None):
                # Add generation id to log message if trace is enabled
                if self.generation:
                    log_message = f"[{self.name}|{self.generation.id}] {message}"

            self.logs += "\n" + log_message
            if level == "error":
                self.logger.error(log_message)
            else:
                self.logger.info(log_message)

    async def validate_input(self, **kwargs):
        return self.input_validator(**kwargs).dict()

    async def validate_output(self, **kwargs):
        return self.output_validator(**kwargs).dict()

    async def run(self, **kwargs) -> Any:
        pass

    async def process_output(self, **kwargs):
        return kwargs
