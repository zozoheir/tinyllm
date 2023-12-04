import os
import unittest

import openai

from tinyllm.functions.llms.lite_llm.lite_llm_chat import LiteLLMChat
from tinyllm.state import States
from tinyllm.tests.base import AsyncioTestCase

openai.api_key = os.environ['OPENAI_API_KEY']


class TestOpenAIChat(AsyncioTestCase):

    def test_openai_chat_script(self):
        test_openai_functions = [
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

        def test_function(asked_property):
            if asked_property == "name":
                return "Elias"
            elif asked_property == "birthday":
                return "January 1st"

        function_callables = {'test_function': test_function}
        litellm_chat = LiteLLMChat(name='Test: LiteLLMChat',
                                   model='gpt-3.5-turbo',
                                   temperature=0,
                                   max_tokens=100,
                                   openai_functions=test_openai_functions,
                                   function_callables=function_callables,
                                   with_memory=True)
        result = self.loop.run_until_complete(litellm_chat(content="What is the user's  birthday?",
                                                           with_functions=True))
        self.assertEqual(litellm_chat.state, States.COMPLETE)


if __name__ == '__main__':
    unittest.main()
