from textwrap import dedent

import numpy as np
import openai

def get_cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_top_n_similar_vectors_index(input_vector, vectors, k=5):
    similarities = [get_cosine_similarity(input_vector, vector) for vector in vectors]
    top_similarities_indices = np.argsort(similarities)[-k:][::-1]
    return [int(index) for index in top_similarities_indices]

def get_openai_embedding(text, model="text-embedding-ada-002"):
    embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    return embedding

def generate_raw_ngrams(text, n):
    tokens = text.split()
    if len(tokens) < n:
        return []
    # Use a list comprehension to generate the n-grams
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    return ngrams

