import unittest

from tinyllm.vector_store import VectorStore, Embeddings


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.store = VectorStore()
        self.store.create_tables()
        self.collection_name = "test_collection"

        # Sample 1
        self.sample_texts = ["text_{}".format(i) for i in range(10)]
        self.sample_metadata = [{"key_{}".format(i): "value_{}".format(i)} for i in range(10)]
        self.store.add_texts(texts=self.sample_texts, metadatas=self.sample_metadata,
                             collection_name=self.collection_name)

        # Sample 2
        self.sample_texts_2 = ["overlap_text_{}".format(i) for i in range(3)]
        self.sample_metadata_2 = [
            {"overlap_key": "overlap_value_1"},
            {"overlap_key": "overlap_value_2"},
            {"overlap_key": "overlap_value_3"},
        ]
        self.store.add_texts(texts=self.sample_texts_2, metadatas=self.sample_metadata_2,
                             collection_name=self.collection_name)

        # Sample 3 with integer metadata values
        self.sample_texts_3 = ["int_text_{}".format(i) for i in range(3)]
        self.sample_metadata_3 = [
            {"int_key": 1},
            {"int_key": 2},
            {"int_key": 3},
        ]
        self.store.add_texts(texts=self.sample_texts_3, metadatas=self.sample_metadata_3,
                             collection_name=self.collection_name)


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

    def test_metadata_filter_on_same_key(self):
        search_text = "overlap_text_0"

        metadata_filter = {"overlap_key": ["overlap_value_1", "overlap_value_2"]}
        results = self.store.similarity_search(query=search_text,
                                               k=1,
                                               collection_filters=self.collection_name,
                                               metadata_filters=metadata_filter)  # Using 10 for k to get all possible matches

        # Check if the results have the expected texts and exclude the one with overlap_value_3
        result_texts = [res['text'] for res in results]
        self.assertIn("overlap_text_0", result_texts)
        self.assertNotIn("overlap_text_1", result_texts)
        self.assertNotIn("overlap_text_2", result_texts)

    def test_metadata_filter_on_int_values(self):
        search_text = "int_text_0"  # Just a placeholder; adjust as necessary based on real data.

        # Use a metadata filter to look for texts that have int_key set to 1 or 2.
        metadata_filter = {"int_key": [1, 2]}
        results = self.store.similarity_search(query=search_text,
                                               k=2,
                                               collection_filters=self.collection_name,
                                               metadata_filters=metadata_filter)

        # Check if the results have the expected texts and exclude the one with int_key = 3
        result_texts = [res['text'] for res in results]
        self.assertIn("int_text_0", result_texts)
        self.assertIn("int_text_1", result_texts)
        self.assertNotIn("int_text_2", result_texts)

    def tearDown(self):
        with self.store._Session() as session:
            session.query(Embeddings).filter_by(collection_name=self.collection_name).delete(synchronize_session=False)
            session.commit()

            for text in self.sample_texts_3:
                session.query(Embeddings).filter_by(text=text).delete(synchronize_session=False)


if __name__ == "__main__":
    unittest.main()
