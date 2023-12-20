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
from tinyllm import langfuse_client, tinyllm_config
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
    processed_output_validator: Optional[Type[Validator]] = None
    is_traced: bool
    evaluators: Optional[list]
    dataset_name: Optional[str]
    stream: Optional[bool]


class DefaultInputValidator(Validator):
    role: Any
    content: Any


class DefaultOutputValidator(Validator):
    response: Any


class DefaultProcessedOutputValidator(Validator):
    response: Any


class Function:

    def __init__(
            self,
            name,
            user_id=None,
            input_validator=Validator,
            output_validator=DefaultOutputValidator,
            processed_output_validator=None,
            evaluators=[],
            dataset_name=None,
            is_traced=True,
            required=True,
            stream=False,
            fallback_strategies={},

    ):
        FunctionInitValidator(
            name=name,
            input_validator=input_validator,
            output_validator=output_validator,
            processed_output_validator=processed_output_validator,
            is_traced=is_traced,
            stream=stream,
        )
        self.parent_observation = None

        self.user = user_id
        self.init_timestamp = dt.datetime.now(pytz.UTC).isoformat()
        self.function_id = str(uuid.uuid4())
        self.logger = getLogger(__name__)
        self.name = name

        self.input_validator = input_validator
        self.output_validator = output_validator
        self.processed_output_validator = processed_output_validator
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
        if is_traced is True:
            self.parent_observation = langfuse_client.trace(CreateTrace(
                name=self.name,
                userId="test")
            )

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



    @fallback_decorator
    async def __call__(self, **kwargs):
        try:
            # Validate input
            self.input = kwargs
            self.transition(States.INPUT_VALIDATION)
            validated_input = self.validate_input(**kwargs)

            # Run
            self.transition(States.RUNNING)
            self.output = await self.run(**validated_input)

            # Validate output
            self.transition(States.OUTPUT_VALIDATION)
            self.output = self.validate_output(**self.output)

            # Process output
            self.transition(States.PROCESSING_OUTPUT)
            self.processed_output = await self.process_output(**self.output)

            # Validate processed output
            self.transition(States.PROCESSED_OUTPUT_VALIDATION)
            if self.processed_output_validator:
                self.processed_output = self.validate_processed_output(**self.processed_output)

            final_output = {"status": "success",
                            "output": self.processed_output}

            # Complete
            self.transition(States.COMPLETE)
            langfuse_client.flush()
            return final_output

        except Exception as e:
            self.error_message = str(e)
            self.transition(States.FAILED, msg=str(e))
            detailed_error_msg = get_exception_info(e)
            self.log(detailed_error_msg, level="error")
            langfuse_client.flush()
            if tinyllm_config['OPS']['DEBUG']:
                raise e
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
        log_level = "error" if new_state == States.FAILED else "info"
        if log_level == 'error':
            self.log(
                f"transition from {self.state} to: {new_state}" + (f" ({msg})" if msg is not None else ""),
                level=log_level,
            )
        else:
            self.log(
                f"transition to: {new_state}" + (f" ({msg})" if msg is not None else ""),
                level=log_level,
            )

        self.state = new_state

    async def evaluate(self,
                       **kwargs):
        generation = kwargs['generation']
        output = kwargs['output']
        run_name = kwargs.get('run_name', "tinyllm_function")
        # Need to link item with generation + call evaluators

        # Create dataset item
        if self.dataset:
            item = langfuse_client.create_dataset_item(
                CreateDatasetItemRequest(dataset_name=self.dataset_name,
                                         input=output,
                                         expected_output="")
            )
            item_client = DatasetItemClient(item, langfuse_client)
            item_client.link(self.generation, run_name)

        # Create eval_data args
        eval_kwargs = {
            'kwargs': kwargs,
            'input': self.input,
            'cache': self.cache,
            'output': self.output,
            'processed_output': self.processed_output,
        }
        # Call evaluators and score
        for evaluator in self.evaluators:
            evaluator_response = await evaluator(generation=generation,
                                                 **eval_kwargs)
            if evaluator_response['status'] != 'success':
                self.log(evaluator_response['message'], level="error")

    def log(self, message, level="info"):

        if tinyllm_config['OPS']['LOGGING']:
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

    def validate_input(self, **kwargs):
        return self.input_validator(**kwargs).dict()

    def validate_output(self, **kwargs):
        return self.output_validator(**kwargs).dict()

    def validate_processed_output(self, **kwargs):
        return self.processed_output_validator(**kwargs).dict()

    async def run(self, **kwargs) -> Any:
        pass

    async def process_output(self, **kwargs):
        return kwargs
