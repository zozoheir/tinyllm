from smartpy.utility.ai_util import get_cosine_similarity
from tinyllm.eval.evaluator import Evaluator


class RetrievalEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run(self, **kwargs):
        truth_context = kwargs["truth_context"]
        question = kwargs["input"]
        retrieved_chunks = kwargs["retrieved_chunks"]
        chunk_texts = [chunk["text"] for chunk in retrieved_chunks]
        chunk_similarities = []
        async def embedding_function(text):
            return [[1]*384] #

        embeddings = await embedding_function(question)
        question_vector = embeddings[0]
        for chunk_text in chunk_texts:
            embeddings = await embedding_function(chunk_text)
            chunk_vector = embeddings[0]
            chunk_similarities.append(get_cosine_similarity(chunk_vector, question_vector))

        embeddings = await embedding_function(truth_context)
        truth_vectors = embeddings[0]
        truth_similarity = get_cosine_similarity(truth_vectors[0], question_vector)
        retrieved_similarity = sum(chunk_similarities) / len(chunk_similarities)

        evals = {
            "truth_similarity": truth_similarity,
            "precision": max(chunk_similarities) / truth_similarity,
            "avg_chunk_similarity_norm": retrieved_similarity / truth_similarity,
            "avg_chunk_similarity": retrieved_similarity,
        }
        return {
            "evals": evals,
            "metadata": {
            }
        }
