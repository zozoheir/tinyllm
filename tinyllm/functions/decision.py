from typing import Union, List

from py2neo import Node, Relationship

from tinyllm import APP
from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator


class DecisionInitValidator(Validator):
    choices: Union[List[str], List[int]]


class DecisionOutputValidator(Validator):
    decision: Union[str, int]


class Decision(Function):
    def __init__(self, choices, **kwargs):
        self.choices = choices
        val = DecisionInitValidator(choices=choices)
        super().__init__(output_validator=DecisionOutputValidator,
                         **kwargs)
