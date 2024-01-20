import unittest

from tinyllm.examples.example_selector import ExampleSelector
from tinyllm.tests.base import AsyncioTestCase


async def embedding_function(text):
    return [[1] * 384]  #


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


        self.example_selector = ExampleSelector(
            name="Test local example selector",
            examples=self.example_texts,
            embedding_function=embedding_function,
        )

        self.loop.run_until_complete(self.example_selector.embed_examples())


    def test_selector(self):
        query = "Find a relevant example"
        results = self.loop.run_until_complete(self.example_selector(input=query,
                                                                           k=1))
        self.assertTrue(len(results['output']['best_examples']) == 1)



if __name__ == '__main__':
    unittest.main()
