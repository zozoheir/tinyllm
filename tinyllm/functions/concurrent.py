import asyncio
from typing import List, Type, Union, Any

from py2neo import Node, Relationship, NodeMatcher

from tinyllm import APP
from tinyllm.state import States
from tinyllm.functions.function import Function, Validator

matcher = NodeMatcher(APP.graph_db)

class ConcurrentValidator(Validator):
    children: List[Union[Function, Type[Function]]]


class ConcurrentInputValidator(Validator):
    inputs: List[Any]


class ConcurrentOutputValidator(Validator):
    outputs: List[Any]


class Concurrent(Function):
    def __init__(self,
                 children: List['Function'] = None,
                 **kwargs):
        m = ConcurrentValidator(children=children,
                              input_validator=ConcurrentInputValidator,
                              output_validator=ConcurrentOutputValidator,
                              **kwargs)

        super().__init__(**kwargs)
        self.children = children if children else []

    async def __call__(self, **kwargs):
        try:
            kwargs = self.handle_inputs(**kwargs)
            self.transition(States.INPUT_VALIDATION)
            kwargs = await self.validate_input(**kwargs)
            self.transition(States.RUNNING)
            tasks = [child.__call__(**kwargs['inputs'][i]) for i, child in enumerate(self.children)]
            output = await asyncio.gather(*tasks)
            self.transition(States.OUTPUT_VALIDATION)
            output = await self.validate_output(output=output)
            self.transition(States.COMPLETE)
            await self.push_to_db()
            return output
        except Exception as e:
            self.handle_exception(e)

    def handle_inputs(self, **kwargs):
        """
        if inputs are not provided, distribute the kwargs to all children
        """
        if 'inputs' not in kwargs.keys():
            return {'inputs':[kwargs for i in range(len(self.children))]}
        else:
            return kwargs

    @property
    def graph_state(self):
        """Returns the state of the current function and all its children."""
        graph_state = {self.name: self.state}
        for child in self.children:
            graph_state.update(child.graph_state)
        return graph_state

    async def push_to_db(self):
        try:
            self.log("Pushing to db")
            included_specifically = APP.config['DB_FUNCTIONS_LOGGING']['DEFAULT'] is True and self.name in \
                                    APP.config['DB_FUNCTIONS_LOGGING']['INCLUDE']
            included_by_default = APP.config['DB_FUNCTIONS_LOGGING']['DEFAULT'] is True and self.name not in \
                                  APP.config['DB_FUNCTIONS_LOGGING']['EXCLUDE']
            if included_specifically or included_by_default:
                attributes_dict = vars(self)
                attributes_dict['class'] = self.__class__.__name__
                attributes_dict = {key: str(value) for key, value in attributes_dict.items()}
                to_ignore = ['input_validator', 'output_validator', 'run_function', 'logger']
                attributes_dict = {str(key): str(value) for key, value in attributes_dict.items() if
                                   value not in to_ignore}
                node = Node(self.name, **attributes_dict)
                APP.graph_db.create(node)

                for child in self.children:
                    self.log(f"Creating relationship between {self.name} and {child.name}")
                    child_node = matcher.match(child.name, function_id=child.function_id).first()
                    relationship = Relationship(node, "CONCURRENT", child_node)
                    relationship['input'] = str(child.output)
                    APP.graph_db.create(relationship)
        except Exception as e:
            self.log(f"Error pushing to db: {e}", level='error')
