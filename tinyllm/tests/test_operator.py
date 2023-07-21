import unittest

from tinyllm.types import Functions
from tinyllm.function import Function


class AddOne(Function):
    def __init__(self, name: str, type: Functions = Functions.OPERATOR, parent_id=None, verbose=True):
        super().__init__(name, type, parent_id, verbose)

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
        add_one = AddOne("AddOne", Functions.OPERATOR, None, True)
        output = await add_one(number=5)
        self.assertEqual(output['number'], 6)



if __name__ == '__main__':
    unittest.main()
