import unittest
import os

import openai

from tinyllm.functions.lite_llm.lite_llm import LiteLLM
from tinyllm.functions.lite_llm.lite_llm_stream import LiteLLMStream
from tinyllm.tests.base import AsyncioTestCase

openai.api_key = os.environ['OPENAI_API_KEY']


class TestlitellmChat(AsyncioTestCase):

    def setUp(self):
        super().setUp()

        self.test_litellm_functions = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "This is the tool you use retrieve ANY information about the user. Use it to answer questions about his birthday and any personal info",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "asked_property": {
                                "type": "string",
                                "enum": ["birthday", "name"],
                                "description": "The specific property the user asked about",
                            },
                        },
                        "required": ["asked_property"],
                    }
                }
            },
        ]

        def function(asked_property):
            if asked_property == "name":
                return "Elias"
            elif asked_property == "birthday":
                return "January 1st"

        self.function_callables = {'test_function': function}

    def test_litellm_chat_stream(self):

        litellmstream_chat = LiteLLMStream(name='Test: LiteLLMChat',
                                           with_memory=True)

        async def get_stream():
            i=0
            async for msg in litellmstream_chat(role='user',
                                                content="What is the user's  birthday?"):
                i = msg
            return i

        result = self.loop.run_until_complete(get_stream())

        self.assertEqual(result['streaming_status'], 'success')

    def test_litellm_chat(self):
        litellm_chat = LiteLLM(name='Test: LiteLLMChat',
                               with_memory=True)
        result = self.loop.run_until_complete(litellm_chat(content="Hi"))


if __name__ == '__main__':
    unittest.main()
