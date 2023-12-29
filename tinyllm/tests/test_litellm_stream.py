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
            msgs = []
            async for msg in litellmstream_chat(messages=[message]):
                i = 0
                msgs.append(msg)
            return msgs

        result = self.loop.run_until_complete(get_stream())
        for res in result:
            print(res['output']['streaming_status'])
            print(res['output']['delta']['content'])
            self.assertEqual(res['status'], 'success')

        deltas = [res['output']['delta']['content'] for res in result]
        final_string = ''.join(deltas)
        self.assertTrue(final_string[-1] != final_string[-2], "The last Delta has been returned twice")



if __name__ == '__main__':
    unittest.main()
