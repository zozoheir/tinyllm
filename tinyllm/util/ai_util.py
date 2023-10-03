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
    text = text.replace("\n", " ")
    try:
        embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    except Exception as e:
        raise e
    return embedding

