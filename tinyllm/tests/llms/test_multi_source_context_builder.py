import unittest

from tinyllm.functions.helpers import count_tokens
from tinyllm.functions.rag.multi_source_context_builder import MultiSourceDocsContextBuilder


class TestMultiSourceDocsContextBuilder(unittest.TestCase):

    def setUp(self):
        super().setUp()

        # Define the initial parameters for MultiSourceDocsContextBuilder
        self.start_string = "KNOWLEDGE GRAPH"
        self.end_string = "KNOWLEDGE GRAPH"

        self.docs_source_1 = [
            {"content": "First document from source 1."},
            {"content": "Second document from source 1."}
        ]

        self.docs_source_2 = [
            {"content": "First document from source 2."},
            {"content": "Second document from source 2."}
        ]
        self.available_token_size = count_tokens(self.docs_source_1)+count_tokens(self.docs_source_2)*0.7
        self.all_docs = [self.docs_source_1, self.docs_source_2]

        self.context_builder = MultiSourceDocsContextBuilder(
            start_string=self.start_string,
            end_string=self.end_string,
            available_token_size=self.available_token_size
        )

    def test_weighted_distribution(self):
        # Provide weights
        weights = [0.5,0.5]

        # Use the MultiSourceDocsContextBuilder to get the final context
        final_context = self.context_builder.get_context(
            docs=self.all_docs,
            header="[post]",
            ignore_keys=[],
            output_format="str",
            weights=weights
        )
        # Assert the presence of the start and end strings in the final context
        self.assertTrue(self.start_string in final_context)
        self.assertTrue(self.end_string in final_context)

        # Approximate check that more content from source 1 is present than source 2
        count_source_1 = sum(doc["content"] in final_context for doc in self.docs_source_1)
        count_source_2 = sum(doc["content"] in final_context for doc in self.docs_source_2)
        self.assertTrue(count_source_1 == count_source_2 == 1)


    def tearDown(self):
        super().tearDown()

