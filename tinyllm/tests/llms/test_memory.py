import unittest

from tinyllm.functions.lite_llm.lite_llm_memory import Memory
from tinyllm.state import States
from tinyllm.tests.base import AsyncioTestCase


class TestOpenAIMemory(AsyncioTestCase):

    def test_openai_memory(self):
        openai_memory = Memory(name="OpenAIMemoryTest")

        # User message
        input_data = {'content': 'Hi agent, how are you?',
                      'role': 'user'}
        result = self.loop.run_until_complete(openai_memory(**input_data))

        self.assertEqual(openai_memory.state, States.COMPLETE)
        self.assertEqual(result['output']['memories'], [])
        self.assertEqual(len(openai_memory.memories), 1)
        self.assertEqual(openai_memory.memories[0], {'role': 'user', 'content': 'Hi agent, how are you?'})


if __name__ == '__main__':
    unittest.main()
