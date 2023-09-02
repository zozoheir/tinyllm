import unittest
import os

from tinyllm.vector_store import VectorStore, Embeddings


class TestVectorStore(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Environment Variables for DB
        self.vector_store = VectorStore()
        self.test_texts = ["Hello, world!", "Hi there!", "How are you?"]
        self.collection_name = 'test_collection'
        self.metadatas = [{"type": "test"}] * len(self.test_texts)

    def test_add_texts(self):

        # Adding test data
        self.vector_store.add_texts(self.test_texts, self.collection_name, self.metadatas)

        query = "Hello, World"
        k = 1
        collection_filters = self.collection_name
        metadata_filters = {"type": "test"}
        results = self.vector_store.similarity_search(query, k, collection_filters, metadata_filters)
        self.assertTrue(len(results) <= k)
        self.assertTrue(all(r['metadata']['type'] == 'test' for r in results))

    @classmethod
    def tearDownClass(self):
        # Remove test data
        with self.vector_store._Session() as session:
            session.query(Embeddings).filter(Embeddings.emetadata['type'].astext == 'test').delete(
                synchronize_session='fetch')
            session.commit()


if __name__ == '__main__':
    unittest.main()
