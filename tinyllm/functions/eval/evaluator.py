from typing import Optional

from tinyllm.function import Function
from tinyllm.validator import Validator



class OutputValidator(Validator):
    evals: dict
    metadata: Optional[dict] = {}


class Evaluator(Function):

    def __init__(self,
                 **kwargs):
        super().__init__(output_validator=OutputValidator,
                         **kwargs)
