import os
import unittest
import asyncio
import os

import openai

from tinyllm.functions.llms.lite_llm.lite_llm_chat import LiteLLMStream, LiteLLM

import openai
from tinyllm.state import States
from tinyllm.tests.base import AsyncioTestCase

openai.api_key = os.environ['OPENAI_API_KEY']


class TestOpenAIChat(AsyncioTestCase):

    def setUp(self):
        super().setUp()

        openai.api_key = os.environ['OPENAI_API_KEY']

        self.test_openai_functions = [
            {
                "name": "test_function",
                "description": "Your default tool. This is the tool you use retrieve ANY information about the user. Use it to answer questions about his birthday and any personal info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "asked_property": {
                            "type": "string",
                            "enum": ["birthday", "name"],
                            "description": "The property the user asked about",
                        },
                    },
                    "required": ["asked_property"],
                }
            }
        ]

        def function(asked_property):
            if asked_property == "name":
                return "Elias"
            elif asked_property == "birthday":
                return "January 1st"

        self.function_callables = {'test_function': function}

    def test_openai_chat_tstream(self):

        litellmstream_chat = LiteLLMStream(name='Test: LiteLLMChat',
                                           model='gpt-3.5-turbo',
                                           temperature=0,
                                           max_tokens=100,
                                           openai_functions=self.test_openai_functions,
                                           function_callables=self.function_callables,
                                           with_memory=True)

        async def get_stream():
            async for api_result, response in litellmstream_chat(content="What is the user's  birthday?",
                                                                 with_functions=False):
                i = 0
            return response

        result = self.loop.run_until_complete(get_stream())

    def test_openai_chat_tstream(self):
        litellmstream_chat = LiteLLM(name='Test: LiteLLMChat',
                                     model='gpt-3.5-turbo',
                                     temperature=0,
                                     max_tokens=100,
                                     openai_functions=self.test_openai_functions,
                                     function_callables=self.function_callables,
                                     with_memory=True)
        result = self.loop.run_until_complete(litellmstream_chat(content="What is the user's  birthday?",
                                                                 with_functions=False))


if __name__ == '__main__':
    unittest.main()
