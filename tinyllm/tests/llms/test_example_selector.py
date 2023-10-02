import unittest

from sqlalchemy import delete

from tinyllm.functions.llms.example_selector import VectorStoreExampleSelector
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.vector_store import VectorStore, Embeddings


class TestExampleSelector(AsyncioTestCase):

    def setUp(self):
        super().setUp()
        self.vector_store = VectorStore()
        self.example_texts = ["This is an example", "Another example", "Yet another example"]
        self.collection_name = 'test_examples'
        self.metadatas = [{"type": "example"}] * len(self.example_texts)
        self.loop.run_until_complete(self.vector_store.add_texts(self.example_texts, self.collection_name, self.metadatas))

        self.example_selector = VectorStoreExampleSelector(name="Test example selector",
                                                           collection_name=self.collection_name)

    def test_selector(self):
        query = "Find a relevant example"
        results = self.loop.run_until_complete(self.example_selector(user_question=query,
                                                                     metadata_filters={"type": ["example"]}))

        self.assertTrue(len(results['best_examples']) <= 1)
        self.assertTrue(all(r['metadata']['type'] == 'example' for r in results['best_examples']))

    def tearDown(self):
        async def clear_db():
            async with self.vector_store._Session() as session:
                await session.begin()
                await session.execute(
                    delete(Embeddings).where(Embeddings.collection_name == self.collection_name)
                )
                await session.commit()

        self.loop.run_until_complete(clear_db())

        super().tearDown()


if __name__ == '__main__':
    unittest.main()
