import os
import unittest

import openai

from tests.base import AsyncioTestCase
from tinyllm.functions.llms.openai.helpers import get_system_message
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat
from tinyllm.types import States

openai.api_key = os.environ['OPENAI_API_KEY']


class TestOpenAIChat(AsyncioTestCase):

    def test_openai_chat_script(self):
        openai_prompt_template = [
            get_system_message("You are an English to Spanish translator")
        ]
        openai_chat = OpenAIChat(name='OpenAI Chat model',
                                 llm_name='gpt-3.5-turbo',
                                 temperature=0,
                                 prompt_template=openai_prompt_template,
                                 n=1)

        result = self.loop.run_until_complete(openai_chat(message="Hello, how are you?"))

        self.assertEqual(openai_chat.state,States.COMPLETE)
        self.assertEqual(result['response'], 'Hola, ¿cómo estás?')

        result = self.loop.run_until_complete(openai_chat(message="Today is Monday"))
        self.assertEqual(openai_chat.state, States.COMPLETE)
        self.assertEqual(len(openai_chat.memory.memories), 4)


if __name__ == '__main__':
    unittest.main()
