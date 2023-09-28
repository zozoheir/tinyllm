import unittest

from tinyllm.functions.llms.context_builder import DocsContextBuilder
from tinyllm.tests.base import AsyncioTestCase


class TestDocsContextBuilder(AsyncioTestCase):

    def setUp(self):
        super().setUp()

        # Define the initial parameters for DocsContextBuilder
        self.start_string = "SUPPORTING DOCS"
        self.end_string = "SUPPORTING DOCS"
        self.available_token_size = 1024  # set some arbitrary limit

        self.docs = [
            {"content": "First document text."},
            {"content": "Second document text, which is slightly longer."},
            {"content": "Third document text."}
        ]

        self.context_builder = DocsContextBuilder(
            name="DocsContextBuilderTest",
            start_string=self.start_string,
            end_string=self.end_string,
            available_token_size=self.available_token_size
        )

    def test_get_context(self):
        # Use the DocsContextBuilder to get the final context
        final_context = self.loop.run_until_complete(self.context_builder(
            docs=self.docs,
            header="[post]",
            ignore_keys=[],
            output_format="str"
        ))

        # Assert the presence of the start and end strings in the final context
        self.assertTrue(self.start_string in final_context['context'])
        self.assertTrue(self.end_string in final_context['context'])

        # Assert the presence of document texts in the final context
        for doc in self.docs:
            self.assertTrue(doc["content"] in final_context['context'])

    def tearDown(self):
        super().tearDown()


if __name__ == '__main__':
    unittest.main()
