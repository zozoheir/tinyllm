import asyncio

from tests.test_function import AddOneOperator
from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator


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


operator = AddOneOperator(name="AddOneTest")

async def my_coroutine():
    raise ValueError("This is a custom exception.")

async def main():
    result = await operator(value=5)
    return result


result = asyncio.run(main())
print(result)