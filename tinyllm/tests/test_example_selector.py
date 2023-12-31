import unittest

from sentence_transformers import SentenceTransformer

from nlp_service.client import NLPService
from rumorz_llms.constants import nlp_service
from tinyllm.examples.example_manager import ExampleManager
from tinyllm.examples.example_selector import ExampleSelector
from tinyllm.util.helpers import get_openai_message
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.tests.base import AsyncioTestCase


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

        async def embedding_function(text):
            res = await nlp_service.get_embeddings(text=text)
            return res

        self.local_example_selector = ExampleSelector(
            name="Test local example selector",
            examples=self.example_texts,
            embedding_function=embedding_function,
        )

        self.loop.run_until_complete(self.local_example_selector.embed_examples())


    def test_selector(self):
        query = "Find a relevant example"
        results = self.loop.run_until_complete(self.local_example_selector(input=query,
                                                                           k=1))
        self.assertTrue(len(results['output']['best_examples']) == 1)



if __name__ == '__main__':
    unittest.main()
