import unittest

from tinyllm.vector_store import VectorStore, Embeddings


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.store = VectorStore()
        self.store.create_tables()
        self.collection_name = "test_collection"
        self.sample_texts = ["text_{}".format(i) for i in range(10)]
        self.sample_metadata = [{"key_{}".format(i): "value_{}".format(i)} for i in range(10)]
        self.store.add_texts(self.sample_texts, self.sample_metadata, self.collection_name)

    def test_inserted_texts_exist(self):
        with self.store._Session() as session:
            texts_in_db = session.query(Embeddings.text).filter_by(collection_name=self.collection_name).all()
            texts_in_db = [t[0] for t in texts_in_db]  # Flatten the result
            for text in self.sample_texts:
                self.assertIn(text, texts_in_db)

    def test_similarity_search(self):
        # Assuming that the 5th text in sample_texts is closest to "search_text". Adjust as needed.
        search_text = self.sample_texts[5] + "_modified"  # This text should closely match with 5th sample_text
        metadata_filter = {"key_5": "value_5"}
        results = self.store.similarity_search(search_text, 1, self.collection_name, metadata_filter)

        # There should be one result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], self.sample_texts[5])
        self.assertDictEqual(results[0]['metadata'], self.sample_metadata[5])

    def tearDown(self):
        with self.store._Session() as session:
            session.query(Embeddings).filter_by(collection_name=self.collection_name).delete(synchronize_session=False)
            session.commit()


if __name__ == "__main__":
    unittest.main()
