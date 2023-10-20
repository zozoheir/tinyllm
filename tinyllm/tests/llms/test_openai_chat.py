import os
import unittest

import openai

from tinyllm.functions.llms.open_ai.openai_chat import OpenAIChat
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.state import States
from tinyllm.tests.base import AsyncioTestCase

openai.api_key = os.environ['OPENAI_API_KEY']

class TestOpenAIChat(AsyncioTestCase):

    def test_openai_chat_script(self):
        openai_prompt_template = OpenAIPromptTemplate(name='OpenAI Prompt Template',
                                                      system_role="You are an English to Spanish translator")

        openai_chat = OpenAIChat(name='Test: OpenAI Chat model',
                                 model='gpt-3.5-turbo',
                                 temperature=0,
                                 prompt_template=openai_prompt_template,
                                 max_tokens=100,
                                 with_memory=True)

        result = self.loop.run_until_complete(openai_chat(message="Hello, how are you?"))

        self.assertEqual(openai_chat.state, States.COMPLETE)
        self.assertEqual(result['response'], 'Hola, ¿cómo estás?')

        result = self.loop.run_until_complete(openai_chat(message="Today is Monday"))
        self.assertEqual(openai_chat.state, States.COMPLETE)
        self.assertEqual(len(openai_chat.memory.memories), 4)


if __name__ == '__main__':
    unittest.main()
