import unittest

from tinyllm.agent.tool.tool import Tool
from tinyllm.rag.document.document import Document
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.agent.tool import Toolkit, tinyllm_toolkit
from tinyllm.eval.evaluator import Evaluator
from tinyllm.memory.memory import BufferMemory
from tinyllm.util.helpers import get_openai_message


# Define the test class
class TestRAG(AsyncioTestCase):

    def test_doc(self):
        doc = Document(content='Hello World')
        doc.to_string()
        #self.assertEqual(result[-1]['status'], 'success', "The last message status should be 'success'")


# This allows the test to be run standalone
if __name__ == '__main__':
    unittest.main()
