import unittest
import os

from tinyllm.tests.base import AsyncioTestCase
from tinyllm.vector_store import VectorStore, Embeddings


class TestVectorStore(AsyncioTestCase):

    @classmethod
    def setUpClass(self):
        # Environment Variables for DB
        self.vector_store = VectorStore()
        self.test_texts = ["Hello, world!", "Hi there!", "How are you?"]
        self.collection_name = 'test_collection'
        self.metadatas = [{"type": "test"}] * len(self.test_texts)

    def test_add_texts(self):

        # Adding test data
        self.loop.run_until_complete(self.vector_store.add_texts(self.test_texts, self.collection_name, self.metadatas))

        query = "Hello, World"
        k = 1
        collection_filters = [self.collection_name]
        metadata_filters = {"type": ["test"]}
        results = self.loop.run_until_complete(self.vector_store.similarity_search(query, k, collection_filters, metadata_filters))
        self.assertTrue(len(results) <= k)
        self.assertTrue(all(r['metadata']['type'] == 'test' for r in results))

    @classmethod
    async def tearDownClass(cls):  # Note the change to `cls` to follow Python convention for class methods
        # Remove test data
        async with cls.vector_store._Session() as session:  # Use an asynchronous session
            await session.begin()
            await session.execute(
                delete(Embeddings).where(Embeddings.emetadata['type'].astext == 'test')
            )
            await session.commit()


if __name__ == '__main__':
    unittest.main()
