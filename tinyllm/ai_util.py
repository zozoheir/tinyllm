import os
import pickle

import numpy as np
import openai

def get_tinyllm_embeddings(library_files, embeddings_path):
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings_dict_list = pickle.load(f)
    else:
        embeddings_dict_list = []
        for file in library_files:
            with open(file, 'r') as f:
                file_content = f.read()
            if len(file_content)>0:
                embeddings_dict_list.append({
                    'file_path': file,
                    'content':file_content,
                    'embeddings': get_embedding(file_content)
                })
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings_dict_list, f)
    return embeddings_dict_list




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

def find_related_files(content, embeddings_dict_list, model="text-embedding-ada-002", top_n=5):
    content_embedding = get_embedding(content, model)
    embeddings_list = [item['embeddings'] for item in embeddings_dict_list]
    similar_embeddings_indices = top_n_similar(content_embedding, embeddings_list, top_n)
    similar_files = [embeddings_dict_list[i] for i in similar_embeddings_indices]
    return similar_files

