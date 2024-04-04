import unittest

from tinyllm.eval.evaluator import Evaluator
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.llms.lite_llm_stream import LiteLLMStream
from tinyllm.util.helpers import get_openai_message
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.util.message import UserMessage, Text


class TestlitellmChat(AsyncioTestCase):

    def setUp(self):
        super().setUp()

    def test_litellm_chat(self):
        message = UserMessage("Hi")
        litellm_chat = LiteLLM(name='Test: LiteLLMChat')
        result = self.loop.run_until_complete(litellm_chat(messages=[message]))
        self.assertEqual(result['status'], 'success')

    def test_litellm_chat_evaluator(self):
        class SuccessFullRunEvaluator(Evaluator):
            async def run(self, **kwargs):
                return {
                    "evals": {
                        "successful_score": 1,
                    },
                    "metadata": {}
                }

        litellm_chat = LiteLLM(name='Test: LiteLLMChat evaluation',
                               run_evaluators=[SuccessFullRunEvaluator()],
                               processed_output_evaluators=[SuccessFullRunEvaluator()])
        message = UserMessage('hi')
        result = self.loop.run_until_complete(litellm_chat(messages=[message]))
        self.assertEqual(result['status'], 'success')



if __name__ == '__main__':
    unittest.main()
