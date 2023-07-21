import unittest

from tinyllm.types import Operators
from tinyllm.operator import Operator


class AddOne(Operator):
    def __init__(self, name: str, operator_type: Operators = Operators.OPERATOR, parent_id=None, verbose=True):
        super().__init__(name, operator_type, parent_id, verbose)

    async def validate_input(self, *args, **kwargs) -> bool:
        if 'number' in kwargs and isinstance(kwargs['number'], int):
            return True
        else:
            return False

    async def get_output(self, *args, **kwargs) -> dict:
        number = kwargs['number']
        return {'number': number + 1}

    async def validate_output(self, *args, **kwargs: dict) -> bool:
        if 'number' in kwargs and isinstance(kwargs['number'], int):
            return True
        else:
            return False



class TestAddOne(unittest.TestCase):

    async def test_add_one(self):
        add_one = AddOne("AddOne", Operators.OPERATOR, None, True)
        output = await add_one(number=5)
        self.assertEqual(output['number'], 6)



if __name__ == '__main__':
    unittest.main()
