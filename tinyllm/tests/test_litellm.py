import unittest

from tinyllm.eval.evaluator import Evaluator
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.llms.lite_llm_stream import LiteLLMStream
from tinyllm.util.helpers import get_openai_message
from tinyllm.tests.base import AsyncioTestCase


class TestlitellmChat(AsyncioTestCase):

    def setUp(self):
        super().setUp()

    def test_litellm_chat_stream(self):
        litellmstream_chat = LiteLLMStream(name='Test: LiteLLM Stream')

        async def get_stream():
            message = get_openai_message(role='user',
                                         content="Hi")
            async for msg in litellmstream_chat(messages=[message]):
                i = 0
            return msg

        result = self.loop.run_until_complete(get_stream())

        self.assertTrue(result['output']['streaming_status'], 'completed')

    def test_litellm_chat(self):
        message = get_openai_message(role='user',
                                     content="Hi")
        litellm_chat = LiteLLM(name='Test: LiteLLMChat')
        result = self.loop.run_until_complete(litellm_chat(messages=[message]))
        self.assertEqual(result['status'], 'success')

    def test_litellm_chat_evaluator(self):
        class SuccessFullRunEvaluator(Evaluator):
            async def run(self, **kwargs):
                print('')
                return {
                    "evals": {
                        "successful_score": 1,
                    },
                    "metadata": {}
                }

        litellm_chat = LiteLLM(name='Test: LiteLLMChat evaluation',
                               run_evaluators=[SuccessFullRunEvaluator()],
                               processed_output_evaluators=[SuccessFullRunEvaluator()])
        message = get_openai_message(role='user',
                                     content="Hi")
        result = self.loop.run_until_complete(litellm_chat(messages=[message]))
        self.assertEqual(result['status'], 'success')


if __name__ == '__main__':
    unittest.main()
