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

    async def push_to_db(self):
        self.log("Pushing to db")
        included_specifically = APP.config['DB_FUNCTIONS_LOGGING']['DEFAULT'] is True and self.name in \
                                APP.config['DB_FUNCTIONS_LOGGING']['INCLUDE']
        included_by_default = APP.config['DB_FUNCTIONS_LOGGING']['DEFAULT'] is True and self.name not in \
                              APP.config['DB_FUNCTIONS_LOGGING']['EXCLUDE']
        if included_specifically or included_by_default:
            node = self.create_function_node()
            APP.graph_db.create(node)
