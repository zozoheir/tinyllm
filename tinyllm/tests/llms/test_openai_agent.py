import asyncio
import os
import unittest

import openai

from tinyllm.functions.llms.open_ai.openai_chat_agent import OpenAIChatAgent
from tinyllm.state import States
from tinyllm.tests.base import AsyncioTestCase

openai.api_key = os.environ['OPENAI_API_KEY']


class TestOpenAIAgent(AsyncioTestCase):

    def test_openai_agent(self):
        test_openai_functions = [
            {
                "name": "test_function",
                "description": "Your default tool. This is the tool you use retrieve ANY information about the user. Use it to answer questions about his birthday and any personal info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "asked_property": {
                            "type": "string",
                            "description": "The property the user asked about",
                        },
                    },
                    "unit": {"type": "string", "enum": ["birthday", "name"]},
                    "required": ["asked_property"],
                },
            }
        ]

        def test_function(asked_property):
            if asked_property == "name":
                return "Elias"
            elif asked_property == "birthday":
                return "January 1st"

        function_callables = {'test_function': test_function}

        openai_agent = OpenAIChatAgent(
            name="Test TinyLLM Agent",
            model="gpt-3.5-turbo",
            openai_functions=test_openai_functions,
            function_callables=function_callables,
            temperature=0,
            max_tokens=1000,
            with_memory=True,
        )

        result = self.loop.run_until_complete(openai_agent(message="Oh nana...what's my name?"))

        self.assertEqual(openai_agent.state, States.COMPLETE)
        self.assertTrue('Elias' in result['response'])


if __name__ == '__main__':
    unittest.main()
