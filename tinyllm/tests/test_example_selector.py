import unittest

from sentence_transformers import SentenceTransformer

from tinyllm.examples.example_manager import ExampleManager
from tinyllm.examples.example_selector import ExampleSelector
from tinyllm.util.helpers import get_openai_message
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.tests.base import AsyncioTestCase

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_function = lambda x: embedding_model.encode(x)


class TestExampleSelector(AsyncioTestCase):

    def setUp(self):
        super().setUp()
        self.example_texts = [
            {
                "user": "Example question",
                "assistant": "Example answer",
            },
            {
                "user": "Another example question",
                "assistant": "Another example answer"
            }
        ]

        self.local_example_selector = ExampleSelector(
            name="Test local example selector",
            examples=self.example_texts,
            embedding_function=embedding_function,
            is_traced=False
        )

    def test_selector(self):
        query = "Find a relevant example"
        results = self.loop.run_until_complete(self.local_example_selector(input=query,
                                                                           k=1))
        self.assertTrue(len(results['output']['best_examples']) == 1)



if __name__ == '__main__':
    unittest.main()
