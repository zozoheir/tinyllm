from typing import List, Dict, Optional, Callable

from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator
from tinyllm.util.ai_util import get_top_n_similar_vectors_index


class InitValidator(Validator):
    collection_name: str

class InputValidator(Validator):
    input: str
    k: Optional[int] = 1

class OutputValidator(Validator):
    best_examples: List[Dict]


class InitValidator(Validator):
    examples: List[dict]
    embedding_function: Callable


class ExampleSelector(Function):
    def __init__(self,
                 examples,
                 embedding_function=None,
                 **kwargs):
        val = InitValidator(examples=examples, embedding_function=embedding_function)
        super().__init__(
            input_validator=InputValidator,
            output_validator=OutputValidator,
            **kwargs
        )
        self.examples = examples
        self.embedding_function = embedding_function
        for example in self.examples:
            if example.get('embeddings') is None and embedding_function is not None:
                example['embeddings'] = self.embedding_function(example['USER'])
            elif example.get('embeddings') is None and embedding_function is None:
                raise Exception('Embedding function is not provided')

        self.embeddings = [example['embeddings'] for example in self.examples]

    async def run(self, **kwargs):
        query_embedding = self.embedding_function(kwargs['input'])
        similar_indexes = get_top_n_similar_vectors_index(input_vector=query_embedding, vectors=self.embeddings, k=kwargs['k'])
        return {'best_examples': [self.examples[i] for i in similar_indexes]}