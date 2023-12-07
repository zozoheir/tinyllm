from typing import Optional, Any

from langfuse.model import CreateScore

from tinyllm.function import Function
from tinyllm.validator import Validator


class EvaluatorInputValidator(Validator):
    generation: Any
    output: dict

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
        for name, score in kwargs['evals'].items():
            kwargs['generation'].score(
                CreateScore(
                    name=name,
                    value=score,
                    comment=kwargs['metadata'],
                )
            )
