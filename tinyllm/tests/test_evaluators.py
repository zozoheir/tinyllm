import unittest

from tinyllm.eval.evaluator import Evaluator
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.llms.lite_llm_stream import LiteLLMStream
from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message
from tinyllm.tests.base import AsyncioTestCase


class TestlitellmChat(AsyncioTestCase):

    def setUp(self):
        super().setUp()

    def test_litellm_chat_evaluator(self):
        class SuccessFullRunEvaluator(Evaluator):
            async def run(self, **kwargs):
                return {
                    "evals": {
                        "successful_score": 1,
                    },
                    "comment": {}
                }

        litellm_chat = LiteLLM(name='Test: LiteLLMChat evaluation',
                               run_evaluators=[SuccessFullRunEvaluator()],
                               processed_output_evaluators=[SuccessFullRunEvaluator(prefix='proc:')])

        message = get_openai_message(role='user',
                                     content="Hi")
        result = self.loop.run_until_complete(litellm_chat(messages=[message]))
        self.assertEqual(result['status'], 'success')

    def test_decorator_eval(self):
        class SuccessFullRunEvaluator(Evaluator):
            async def run(self, **kwargs):
                return {
                    "evals": {
                        "successful_score": 1,
                    },
                    "comment": {}
                }

        @observation(observation_type='span', evaluators=[SuccessFullRunEvaluator(prefix='decor:')])
        async def function():
            return {
                'nothing': 'here'
            }

        self.loop.run_until_complete(function())




if __name__ == '__main__':
    unittest.main()
