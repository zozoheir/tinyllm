import unittest

from sentence_transformers import SentenceTransformer
from sqlalchemy import delete

from tinyllm.tests.base import AsyncioTestCase
from tinyllm.rag.vector_store import VectorStore, Embeddings

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_function = lambda x: embedding_model.encode(x)


class TestVectorStore(AsyncioTestCase):

    def setUp(self):
        super().setUp()
        # Environment Variables for DB
        self.vector_store = VectorStore(embedding_function=embedding_function)
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
        docs = results['documents']
        self.assertTrue(len(docs) <= k)
        self.assertTrue(all(r['document'].metadata['type'] == 'test' for r in docs))


    def tearDown(self):

        async def clear_dbb():
            async with self.vector_store._Session() as session:  # Use an asynchronous session
                await session.begin()
                await session.execute(
                    delete(Embeddings).where(Embeddings.collection_name == self.collection_name)
                )
                await session.commit()

        self.loop.run_until_complete(clear_dbb())

        super().tearDown()

if __name__ == '__main__':
    unittest.main()
