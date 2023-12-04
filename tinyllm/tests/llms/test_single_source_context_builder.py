import unittest

from tinyllm.functions.llms.util.single_source_context_builder import SingleSourceDocsContextBuilder


class TestDocsContextBuilder(unittest.TestCase):

    def setUp(self):
        super().setUp()

        # Define the initial parameters for DocsContextBuilder
        self.start_string = "KNOWLEDGE GRAPH"
        self.end_string = "KNOWLEDGE GRAPH"
        self.available_token_size = 1024  # set some arbitrary limit

        self.docs = [
            {"content": "First document text.", "test": "test"},
            {"content": "Second document text, which is slightly longer.", "test": "test"},
            {"content": "Third document text.",
             "test": "test"}
        ]

        self.context_builder = SingleSourceDocsContextBuilder(
            start_string=self.start_string,
            end_string=self.end_string,
            available_token_size=self.available_token_size
        )

    def test_get_context(self):
        # Use the DocsContextBuilder to get the final context
        final_context = self.context_builder.get_context(
            docs=self.docs,
            header="[post]",
            ignore_keys=['test'],
            output_format="str"
        )

        # Assert the presence of the start and end strings in the final context
        self.assertTrue(self.start_string in final_context)
        self.assertTrue(self.end_string in final_context)
        self.assertTrue("test" not in final_context)
        # Assert the presence of document texts in the final context
        for doc in self.docs:
            self.assertTrue(doc["content"] in final_context)

    def tearDown(self):
        super().tearDown()


if __name__ == '__main__':
    unittest.main()
