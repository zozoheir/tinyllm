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
from typing import Any, Callable, Optional, Type, Dict
import pytz
from langfuse.client import DatasetItemClient

from smartpy.utility.log_util import getLogger
from smartpy.utility.py_util import get_exception_info
from tinyllm.exceptions import InvalidStateTransition
from tinyllm.llm_ops import LLMTrace, langfuse_client, LLMDataset
from tinyllm.state import States, ALLOWED_TRANSITIONS
from tinyllm.functions.validator import Validator
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
    evaluators: Optional[list]
    dataset_name: Optional[str]


class DefaultInputValidator(Validator):
    input: str


class DefaultOutputValidator(Validator):
    response: str


class Function:

    def __init__(
            self,
            name,
            user_id=None,
            input_validator=Validator,
            output_validator=DefaultOutputValidator,
            evaluators=[],
            dataset: LLMDataset = LLMDataset(name='tinyllm'),
            is_traced=True,
            required=True,
            llm_trace: LLMTrace = None,
            fallback_strategies={},

    ):
        w = FunctionInitValidator(
            name=name,
            input_validator=input_validator,
            output_validator=output_validator,
            is_traced=is_traced,
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
        self.llm_trace = None
        if llm_trace is None and is_traced is True:
            self.llm_trace = LLMTrace(
                name=self.name,
                userId="test",
            )
        else:
            self.llm_trace = llm_trace
        self.cache = {}
        self.evaluators = evaluators
        self.dataset = dataset
        self.fallback_strategies = fallback_strategies

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
            self.output = await self.process_output(**self.output)
            self.processed_output = self.output
            if self.evaluators:
                self.transition(States.EVALUATING)
                await self.evaluate(**kwargs)
            self.transition(States.COMPLETE)
            return {"status": "success",
                    "output": self.output}
        except Exception as e:
            self.error_message = str(e)
            self.transition(States.FAILED, msg=str(e))
            detailed_error_msg = get_exception_info(e)
            self.log(detailed_error_msg, level="error")
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
        eval_data = self.cache
        if self.processed_output:
            eval_data.update(self.processed_output)
        if kwargs:
            eval_data.update(kwargs)

        if self.dataset:
            item = self.dataset.create_item(
                input=kwargs.get('input', "None"),
                expected_output=kwargs.get('expected_output', "None"),
            )
            item_client = DatasetItemClient(item, langfuse_client)
            item_client.link(self.llm_trace.current_generation, kwargs.get('run_name', "None"))

        for evaluator in self.evaluators:
            evaluator_response = await evaluator(**eval_data)
            if evaluator_response['status'] == 'success':
                for name, value in evaluator_response['output']['evals'].items():
                    self.llm_trace.score_generation(
                        name=name,
                        value=value,
                        comment=kwargs.get('score_comment', "None"),
                    )
            else:
                self.log(evaluator_response['message'], level="error")

    def log(self, message, level="info"):
        log_message = f"[{self.name}] {message}"
        if getattr(self, 'llm_trace', None):
            if self.llm_trace.current_generation:
                log_message = f"[{self.name}|{self.llm_trace.current_generation.id}] {message}"

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
