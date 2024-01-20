from typing import List, Dict, Optional, Callable

import numpy as np

from tinyllm.function import Function
from tinyllm.validator import Validator
from tinyllm.util.ai_util import get_top_n_similar_vectors_index


class ExampleSelectorInitValidator(Validator):
    collection_name: str

class InputValidator(Validator):
    input: str
    k: Optional[int] = 1

class OutputValidator(Validator):
    best_examples: List[Dict]

class ProcessedOutputValidator(Validator):
    best_examples: List[Dict]

class ExampleSelectorInitValidator(Validator):
    examples: List[dict]
    embedding_function: Callable
    embeddings: Optional[List]


class ExampleSelector(Function):

    def __init__(self,
                 embedding_function,
                 examples=[],
                 embeddings=None,
                 **kwargs):
        ExampleSelectorInitValidator(examples=examples,
                                     embedding_function=embedding_function,
                                     embeddings=embeddings)
        super().__init__(
            input_validator=InputValidator,
            output_validator=OutputValidator,
            processed_output_validator=ProcessedOutputValidator,
            **kwargs
        )
        self.example_dicts = examples
        self.embeddings = embeddings
        self.embedding_function = embedding_function
        all_example_dicts_have_embeddings = all([example.get('embedding') is not None for example in self.example_dicts])
        if all_example_dicts_have_embeddings is False and self.embedding_function is None:
            raise Exception('Example selector needs either an embedding function or vector embeddings for each example')

    async def embed_examples(self,
                             **kwargs):
        example_dicts = kwargs.get('example_dicts', self.example_dicts)
        embeddings = []
        for example in example_dicts:
            embeddings_list = await self.embedding_function(example['user'])
            embeddings.append(embeddings_list[0])
        self.embeddings = embeddings

    async def run(self, **kwargs):
        embeddings = await self.embedding_function(kwargs['input'])
        similar_indexes = get_top_n_similar_vectors_index(input_vector=embeddings[0], vectors=self.embeddings, k=kwargs['k'])
        return {'best_examples': [self.example_dicts[i] for i in similar_indexes]}

    async def process_output(self, **kwargs):
        result = kwargs['best_examples']
        return {'best_examples': result}