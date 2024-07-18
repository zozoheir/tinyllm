import unittest

from tinyllm.llms.lite_llm_stream import LiteLLMStream
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.util.message import UserMessage


class TestlitellmChat(AsyncioTestCase):

    def setUp(self):
        super().setUp()

    def test_litellm_chat_stream(self):
        litellmstream_chat = LiteLLMStream(name='Test: LiteLLM Stream')

        async def get_stream():
            message = UserMessage('Hi')
            msgs = []
            async for msg in litellmstream_chat(messages=[message]):
                i = 0
                msgs.append(msg)
            return msgs

        result = self.loop.run_until_complete(get_stream())
        deltas = []
        for res in result:
            print(res['output']['streaming_status'])
            if res['output']['streaming_status'] != 'finished-streaming':
                deltas.append(res['output']['last_completion_delta']['content'])

            self.assertEqual(res['status'], 'success')
        final_string = ''.join(deltas)
        self.assertTrue(final_string[-1] != final_string[-2], "The last Delta has been returned twice")



if __name__ == '__main__':
    unittest.main()
