import unittest

from tinyllm.functions.eval.evaluator import Evaluator
from tinyllm.functions.llms.lite_llm import LiteLLM
from tinyllm.functions.llms.lite_llm_stream import LiteLLMStream
from tinyllm.functions.util.helpers import get_openai_message
from tests.base import AsyncioTestCase


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
            print('evaluator')
            async def run(self, **kwargs):
                return {
                    "evals": {
                        "successful_score": 1,
                    },
                    "metadata": {}
                }

        litellm_chat = LiteLLM(name='Test: LiteLLMChat evaluation',
                               evaluators=[
                                   SuccessFullRunEvaluator(
                                       name="Successful run evaluator",
                                       is_traced=False,
                                   )])
        message = get_openai_message(role='user',
                                     content="Hi")
        result = self.loop.run_until_complete(litellm_chat(messages=[message]))
        self.assertEqual(result['status'], 'success')


if __name__ == '__main__':
    unittest.main()
