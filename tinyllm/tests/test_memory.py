import unittest

from tinyllm.memory.memory import BufferMemory
from tinyllm.state import States
from tinyllm.tests.base import AsyncioTestCase


class TestMemory(AsyncioTestCase):

    def test_memory(self):
        memory = BufferMemory(name="Memory test")

        # User message
        msg = {
            'content': 'Hi agent, how are you?',
            'role': 'user'
        }

        result = self.loop.run_until_complete(memory(message=msg))

        self.assertEqual(memory.state, States.COMPLETE)
        self.assertEqual(len(memory.memories), 1)
        self.assertEqual(memory.memories[0], {'role': 'user', 'content': 'Hi agent, how are you?'})


if __name__ == '__main__':
    unittest.main()
