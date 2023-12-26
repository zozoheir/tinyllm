import pprint
from typing import Optional, Any, Type, Union
from tinyllm.function import Function
from tinyllm.validator import Validator



class EvaluatorInitValidator(Validator):
    prefix: Optional[str] = ''

class EvaluatorInputValidator(Validator):
    observation: Any

class EvaluatorInputValidator(Validator):
    observation: Any

class EvaluatorOutputValidator(Validator):
    evals: dict
    comment: Any


class Evaluator(Function):

    def __init__(self,
                 prefix='',
                 **kwargs):
        EvaluatorInitValidator(prefix=prefix)
        super().__init__(output_validator=EvaluatorOutputValidator,
                         input_validator=EvaluatorInputValidator,
                         **kwargs)
        self.prefix = prefix

    async def process_output(self, **kwargs):
        for name, score in kwargs['evals'].items():
            self.input['observation'].score(
                name=self.prefix+name,
                value=score,
                comment=pprint.pformat(kwargs.get('metadata',{})),
            )
        return kwargs