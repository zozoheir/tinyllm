import unittest
from tests.base import AsyncioTestCase
from tinyllm.functions.llms.openai.openai_memory import OpenAIMemory
from tinyllm.types import States


class TestOpenAIMemory(AsyncioTestCase):

    def test_openai_memory(self):
        memory_operator = OpenAIMemory(name="OpenAIMemoryTest")

        # User message
        input_data = {'message': 'Hi agent, how are you?',
                      'role': 'user'}
        result = self.loop.run_until_complete(memory_operator(**input_data))

        self.assertEqual(memory_operator.state, States.COMPLETE)
        self.assertEqual(result, {'success': True})
        self.assertEqual(memory_operator.memories[0], {'role': 'user', 'content': 'Hi agent, how are you?'})


        # assistant response
        input_data = {'message': 'I am good, thanks for asking. And you?',
                      'role': 'assistant'}
        result = self.loop.run_until_complete(memory_operator(**input_data))
        self.assertEqual(memory_operator.state, States.COMPLETE)
        self.assertEqual(result, {'success': True})
        self.assertEqual(memory_operator.memories[1], {'role': 'assistant', 'content': 'I am good, thanks for asking. And you?'})




if __name__ == '__main__':
    unittest.main()
