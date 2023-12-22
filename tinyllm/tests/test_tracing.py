import unittest

from tinyllm.eval.evaluator import Evaluator
from tinyllm.function import Function
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.llms.lite_llm_stream import LiteLLMStream
from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message
from tinyllm.tests.base import AsyncioTestCase


class TestlitellmChat(AsyncioTestCase):

    def setUp(self):
        super().setUp()

    def test_function_tracing(self):

        class TestFunction(Function):

            @observation(observation_type='span')
            async def run(self, **kwargs):
                return {
                    "result": 1
                }

            @observation(observation_type='span')
            async def process_output(self, **kwargs):
                result = 10 + kwargs['result']
                return {
                    "result": result,
                }

        test_func = TestFunction(name='Test: tracing')
        message = {
            'role': 'user',
            'content': "Hi"
        }
        result = self.loop.run_until_complete(test_func(message=message))
        self.assertEqual(result['status'], 'success')


if __name__ == '__main__':
    unittest.main()
