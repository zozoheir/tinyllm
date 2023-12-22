import unittest

from tinyllm.eval.evaluator import Evaluator
from tinyllm.function import Function
from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message
from tinyllm.tests.base import AsyncioTestCase


class SuccessFullRunEvaluator(Evaluator):
    async def run(self, **kwargs):
        print('')
        return {
            "evals": {
                "successful_score": 1,
            },
            "metadata": {}
        }


class TestEvaluators(AsyncioTestCase):


    def test_evaluator(self):


        litellm_chat = Function(name='Test: LiteLLMChat evaluation',
                                run_evaluators=[SuccessFullRunEvaluator()],
                                processed_output_evaluators=[SuccessFullRunEvaluator()])
        message = get_openai_message(role='user',
                                     content="Hi")
        result = self.loop.run_until_complete(litellm_chat(messages=[message]))
        self.assertEqual(result['status'], 'success')


    def test_evaluator_decorator(self):

        @observation('span', evaluators=[SuccessFullRunEvaluator()])
        async def run_func(something=None):
            return {
                'status': 'success',
            }

        result = self.loop.run_until_complete(run_func(something='something'))

if __name__ == '__main__':
    unittest.main()
