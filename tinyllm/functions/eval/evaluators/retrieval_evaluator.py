from smartpy.utility.ai_util import get_cosine_similarity
from tinyllm import default_embedding_model
from tinyllm.functions.eval.evaluator import Evaluator


class RetrievalEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run(self, **kwargs):
        truth_context = kwargs["truth_context"]
        question = kwargs["input"]
        retrieved_chunks = kwargs["retrieved_chunks"]
        chunk_texts = [chunk["text"] for chunk in retrieved_chunks]
        chunk_similarities = []

        question_vector = default_embedding_model(question)
        for chunk_text in chunk_texts:
            chunk_vector = default_embedding_model(chunk_text)
            chunk_similarities.append(get_cosine_similarity(chunk_vector, question_vector))

        truth_vector = default_embedding_model(truth_context)
        truth_similarity = get_cosine_similarity(truth_vector, question_vector)
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
