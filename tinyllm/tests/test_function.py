import unittest

from tinyllm.tests.base import AsyncioTestCase
from tinyllm.function import Function
from tinyllm.exceptions import InvalidStateTransition
from tinyllm.validator import Validator
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
        self.assertEqual(result['output']["value"], 6.0)

    def test_invalid_state_transition(self):
        operator = AddOneOperator(name="AddOneTest")
        with self.assertRaises(InvalidStateTransition):
            operator.transition(States.COMPLETE)

    def test_invalid_input(self):
        operator = AddOneOperator()
        self.loop.run_until_complete(operator(value="wrong input"))
        assert operator.state == States.FAILED



if __name__ == '__main__':
    unittest.main()
