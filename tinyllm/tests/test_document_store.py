import unittest

from tinyllm.rag.document.document import Document
from tinyllm.rag.document.store import DocumentStore


class TestDocsStore(unittest.TestCase):

    def setUp(self):
        # Define the initial parameters for DocsContextBuilder
        self.docs = [
            {"content": "First document text.",
             "metadata": {}},
            {"content": "Second document text, which is slightly longer.",
             "metadata": {}},
            {"content": "Third document text.",
             "metadata": {}}
        ]
        self.docs = [Document(**doc) for doc in self.docs]
        self.document_store = DocumentStore()
        self.document_store.add_docs(
            name='test_section',
            docs=self.docs
        )
        self.docs_2 = [
            {"content": "2 First document text.",
             "metadata": {}},
            {"content": "2 Second document text, which is slightly longer.",
             "metadata": {}},
            {"content": "2 Third document text.",
             "metadata": {}}
        ]
        self.docs_2 = [Document(**doc,
                                header="[doc]",
                                include_keys=['content','metadata']) for doc in self.docs_2]
        self.document_store.add_docs(
            name='test_section_2',
            docs=self.docs_2
        )

    def test_get_context(self):
        start_string = "SUPPORTING DOCS"
        end_string = "END OF SUPPORTING DOCS"

        # We limit the context size to the size of the first doc source in the store, + 1 out of 3 docs from the second source
        doc1_size = sum([i.size for i in self.docs])
        context_available = doc1_size + self.docs_2[0].size

        # Use the DocsContextBuilder to get the final context
        final_context = self.document_store.to_string(
            start_string=start_string,
            end_string=end_string,
            context_size=context_available,
            weights=[0.6, 0.4]
        )
        # Assert the presence of the start and end strings in the final context
        self.assertTrue(start_string in final_context)
        self.assertTrue(end_string in final_context)
        # Assert the presence of document texts in the final context
        for doc in self.docs[:2]:
            self.assertTrue(doc.to_string() in final_context)
        for doc in self.docs_2[:1]:
            self.assertTrue(doc.to_string() in final_context)


if __name__ == '__main__':
    unittest.main()
