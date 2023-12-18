DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Lazy loading function
def load_embedding_model(model, **kwargs):
    if not hasattr(load_embedding_model, "model"):
        from sentence_transformers import SentenceTransformer
        load_embedding_model.model = SentenceTransformer(
            model,
            **kwargs
        )
    return load_embedding_model.model

# Function to get embeddings
def get_sentence_embeddings(model=DEFAULT_EMBEDDING_MODEL, text=None):
    model = load_embedding_model(model)
    return model.encode(text)
