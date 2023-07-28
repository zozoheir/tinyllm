from typing import List, Union, Dict, Type

from py2neo import Node, Relationship
from py2neo import Graph, NodeMatcher

from tinyllm import APP
from tinyllm.functions.function import Function
from tinyllm.state import States
from tinyllm.functions.validator import Validator

matcher = NodeMatcher(APP.graph_db)

class ChainValidator(Validator):
    children: List[Union[Function, Type[Function]]]


class Chain(Function):

    def __init__(self,
                 children,
                 **kwargs):
        super().__init__(output_validator=Validator,
                         **kwargs)
        self.children = children if children else []

    async def __call__(self, **kwargs):
        try:
            self.transition(States.INPUT_VALIDATION)
            kwargs = await self.validate_input(**kwargs)
            self.transition(States.RUNNING)
            output = None
            for child in self.children:
                output = await child(**kwargs)
                kwargs = output
            self.transition(States.OUTPUT_VALIDATION)
            output = await self.validate_output(**output)
            self.transition(States.COMPLETE)

            try:
                await self.push_to_db()
            except Exception as e:
                self.log(f"Error pushing to db: {e}", level='error')
            return output
        except Exception as e:
            self.error_message = str(e)
            self.transition(States.FAILED,
                            msg=str(e))
            try:
                await self.push_to_db()
            except Exception as e:
                self.log(f"Error pushing to db: {e}", level='error')

    @property
    def graph_state(self):
        """Returns the state of the current function and all its children."""
        graph_state = {self.name: self.state}
        for child in self.children:
            graph_state.update(child.graph_state)
        return graph_state


    async def push_to_db(self):
        self.log("Pushing to db")
        included_specifically = APP.config['DB_FUNCTIONS_LOGGING']['DEFAULT'] is True and self.name in \
                                APP.config['DB_FUNCTIONS_LOGGING']['INCLUDE']
        included_by_default = APP.config['DB_FUNCTIONS_LOGGING']['DEFAULT'] is True and self.name not in \
                              APP.config['DB_FUNCTIONS_LOGGING']['EXCLUDE']
        if included_specifically or included_by_default:
            node = self.create_function_node()
            APP.graph_db.create(node)
            next_node = matcher.match(self.children[0].name, function_id=self.children[0].function_id).first()
            relationship = Relationship(node, "CALLS", next_node)
            relationship['input'] = str(self.output)
            APP.graph_db.create(relationship)

            for i in range(len(self.children) - 1):
                current_node = matcher.match(self.children[i].name, function_id=self.children[i].function_id).first()
                next_node = matcher.match(self.children[i+1].name, function_id=self.children[i + 1].function_id).first()
                relationship = Relationship(current_node, "CALLS", next_node)
                relationship['input'] = str(self.children[i].output)
                APP.graph_db.create(relationship)
