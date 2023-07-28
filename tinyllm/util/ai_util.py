import numpy as np
import openai

def get_cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def top_n_similar(input_vector, vector_list, top_n=5):
    similarities = [get_cosine_similarity(input_vector, vector) for vector in vector_list]
    top_similarities_indices = np.argsort(similarities)[-top_n:][::-1]
    return top_similarities_indices

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    try:
        embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    except Exception as e:
        raise e
    return embedding

