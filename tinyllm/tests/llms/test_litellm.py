import unittest
import os

import openai

from tinyllm.functions.eval.evaluator import Evaluator
from tinyllm.functions.lite_llm.lite_llm import LiteLLM
from tinyllm.functions.lite_llm.lite_llm_stream import LiteLLMStream
from tinyllm.tests.base import AsyncioTestCase

openai.api_key = os.environ['OPENAI_API_KEY']


class TestlitellmChat(AsyncioTestCase):

    def test_litellm_chat_stream(self):

        litellmstream_chat = LiteLLMStream(name='Test: LiteLLM Stream',
                                           with_memory=True)

        async def get_stream():
            async for msg in litellmstream_chat(role='user',
                                                content="What is the user's  birthday?"):
                if msg['status'] == 'success':
                    if msg['output']['streaming_status'] == 'completed':
                        return msg

        result = self.loop.run_until_complete(get_stream())

        self.assertTrue(result['output']['streaming_status'], 'completed')

    def test_litellm_chat(self):
        litellm_chat = LiteLLM(name='Test: LiteLLMChat',
                               with_memory=True)
        result = self.loop.run_until_complete(litellm_chat(content="Hi"))
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
                               evaluators=[SuccessFullRunEvaluator(
                                   name="Successful run evaluator",
                                   is_traced=False,
                               )],
                               with_memory=True)

        result = self.loop.run_until_complete(litellm_chat(content="Hi"))
        self.assertEqual(result['status'], 'success')


if __name__ == '__main__':
    unittest.main()
