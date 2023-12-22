from typing import Optional, Any, Type, Union

from langfuse.client import StatefulClient, StatefulSpanClient, StatefulGenerationClient

from tinyllm.function import Function
from tinyllm.validator import Validator


class EvaluatorInputValidator(Validator):
    observation: Any


class EvaluatorOutputValidator(Validator):
    evals: dict
    metadata: Optional[dict] = {}


class Evaluator(Function):

    def __init__(self,
                 **kwargs):
        super().__init__(output_validator=EvaluatorOutputValidator,
                         input_validator=EvaluatorInputValidator,
                         **kwargs)

    async def process_output(self, **kwargs):
        print('evaluator process_output')
        for name, score in kwargs['evals'].items():
            self.input['observation'].score(
                name=name,
                value=score,
                comment=str(kwargs['metadata']),
            )
