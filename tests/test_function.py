import unittest

from tests.base import AsyncioTestCase
from tinyllm import APP
from tinyllm.functions.function import Function
from tinyllm.exceptions import InvalidStateTransition
from tinyllm.functions.validator import Validator
from tinyllm.state import States


class InputValidator(Validator):
    value: float

class OutputValidator(Validator):
    value: float

class AddOneOperator(Function):
    def __init__(self, **kwargs):
        super().__init__(input_validator=InputValidator, output_validator=OutputValidator, **kwargs)

    async def run(self, **kwargs):
        value = kwargs["value"]
        self.log("Adding one to {}".format(value))
        result = value + 1
        self.log("Done adding")
        return {"value": result}


class TestFunction(AsyncioTestCase):

    def test_add_one(self):
        operator = AddOneOperator(name="AddOneTest")
        result = self.loop.run_until_complete(operator(value=5.0))
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], 6.0)

    def test_invalid_state_transition(self):
        operator = AddOneOperator(name="AddOneTest")
        with self.assertRaises(InvalidStateTransition):
            operator.transition(States.COMPLETE)

    def test_invalid_input(self):
        operator = AddOneOperator(name="AddOneTest")
        self.loop.run_until_complete(operator(value="wrong input"))
        assert operator.state == States.FAILED


    def test_push_to_db(self):
        async def dummy_function(**kwargs):
            return kwargs
        test_function = Function(name='test_function',
                                 run_function=dummy_function,
                                 parent_id=None,
                                 verbose=True)
        self.loop.run_until_complete(test_function())

        node = APP.graph_db.evaluate(f"MATCH (n:{test_function.name}) WHERE n.class = $name RETURN n",
                                     parameters={"name": test_function.__class__.__name__})

        self.assertIsNotNone(node, "Node was not created in the database")

        APP.graph_db.run(f"MATCH (n:{test_function.__class__.__name__}) WHERE n.name = $name DETACH DELETE n",
                         parameters={"name": test_function.name})

        node = APP.graph_db.evaluate(f"MATCH (n:{test_function.name}) WHERE n.class = $name RETURN n",
                                     parameters={"name": test_function.name})

        self.assertIsNone(node, "Node was not properly deleted from database")


if __name__ == '__main__':
    unittest.main()
