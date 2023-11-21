import unittest

from tinyllm.functions.function import Function
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.util.fallback_strategy import KwargsChangeStrategy


class CustomException(Exception):
    pass

strategies = {
    CustomException: KwargsChangeStrategy(fallback_kwargs={'text':'nono'}),
}

class TestFunction(Function):

    def __init__(self,
                 **kwargs):
        super().__init__(
            fallback_strategies=strategies,
            **kwargs
        )

    async def run(self, **kwargs):
        return {'response': kwargs['text']}

    async def process_output(self, **kwargs):
        if kwargs['response']=='exception':
            raise CustomException('This is a custom exception')
        else:
            return {
                'response': 'success'
            }


test_function = TestFunction(name='test fallback decorator')


class TestFallbackDecorator(AsyncioTestCase):

    def test_run_fallback(self):
        result = self.loop.run_until_complete(test_function(text='exception'))
        self.assertTrue(result['output']['response'] == 'success')
