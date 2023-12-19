from typing import Optional, Any, Type, Union

from langfuse.client import StatefulClient, StatefulSpanClient, StatefulGenerationClient
from langfuse.model import CreateScore

from tinyllm.function import Function
from tinyllm.validator import Validator


class EvaluatorInputValidator(Validator):
    observation: Union[StatefulGenerationClient, StatefulSpanClient]
    output: Any
    function: Any


class EvaluatorOutputValidator(Validator):
    evals: dict
    metadata: Optional[dict] = {}


class Evaluator(Function):

    def __init__(self,
                 is_traced: bool = False,
                 **kwargs):
        super().__init__(output_validator=EvaluatorOutputValidator,
                         input_validator=EvaluatorInputValidator,
                         is_traced=is_traced,
                                   ** kwargs)

    async def process_output(self, **kwargs):
        for name, score in kwargs['evals'].items():
            self.input['observation'].score(
                CreateScore(
                    name=name,
                    value=score,
                    comment=str(kwargs['metadata']),
                )
            )
