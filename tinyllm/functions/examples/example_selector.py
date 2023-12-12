from typing import List, Dict, Optional, Callable

from tinyllm import default_embedding_model
from tinyllm.function import Function
from tinyllm.validator import Validator
from tinyllm.util.ai_util import get_top_n_similar_vectors_index


class InitValidator(Validator):
    collection_name: str

class InputValidator(Validator):
    input: str
    k: Optional[int] = 1

class OutputValidator(Validator):
    best_examples: List[Dict]

class ProcessedOutputValidator(Validator):
    best_examples: List[Dict]

class InitValidator(Validator):
    examples: List[dict]
    embedding_function: Callable


class ExampleSelector(Function):
    def __init__(self,
                 examples=[],
                 embedding_function=default_embedding_model,
                 **kwargs):
        val = InitValidator(examples=examples, embedding_function=embedding_function)
        super().__init__(
            input_validator=InputValidator,
            output_validator=OutputValidator,
            processed_output_validator=ProcessedOutputValidator,
            **kwargs
        )
        self.example_dicts = examples
        self.embedding_function = embedding_function
        for example in self.example_dicts:
            if example.get('embeddings') is None and embedding_function is not None:
                example['embeddings'] = self.embedding_function(example['USER'])
            elif example.get('embeddings') is None and embedding_function is None:
                raise Exception('Example selector needs embedding function or existing embeddings to work')

        self.embeddings = [example['embeddings'] for example in self.example_dicts]

    async def run(self, **kwargs):
        query_embedding = self.embedding_function(kwargs['input'])
        similar_indexes = get_top_n_similar_vectors_index(input_vector=query_embedding, vectors=self.embeddings, k=kwargs['k'])
        return {'best_examples': [self.example_dicts[i] for i in similar_indexes]}

    async def process_output(self, **kwargs):
        result = kwargs['best_examples']
        return {'best_examples': result}