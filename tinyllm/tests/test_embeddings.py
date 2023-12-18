import unittest

import numpy as np

from tinyllm.embeddings.models import get_sentence_embeddings


# Create a unittest case:

class TestEmbeddings(unittest.TestCase):
    def test_get_sentence_embeddings(self):
        embeddings = get_sentence_embeddings(text='This is a test')
        self.assertTrue(type(embeddings) == np.ndarray)