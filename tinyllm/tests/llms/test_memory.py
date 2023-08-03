import unittest
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.functions.llms.open_ai.openai_memory import OpenAIMemory
from tinyllm.state import States


class TestOpenAIMemory(AsyncioTestCase):

    def test_openai_memory(self):
        openai_memory = OpenAIMemory(name="OpenAIMemoryTest")

        # User message
        input_data = {'content': 'Hi agent, how are you?',
                      'role': 'user'}
        result = self.loop.run_until_complete(openai_memory(openai_message=input_data))

        self.assertEqual(openai_memory.state, States.COMPLETE)
        self.assertEqual(result, {'success': True})
        self.assertEqual(openai_memory.memories[0], {'role': 'user', 'content': 'Hi agent, how are you?'})


        # assistant response
        input_data = {'content': 'I am good, thanks for asking. And you?',
                      'role': 'assistant'}
        result = self.loop.run_until_complete(openai_memory(openai_message=input_data))
        self.assertEqual(openai_memory.state, States.COMPLETE)
        self.assertEqual(result, {'success': True})
        self.assertEqual(openai_memory.memories[1], {'role': 'assistant', 'content': 'I am good, thanks for asking. And you?'})




if __name__ == '__main__':
    unittest.main()
