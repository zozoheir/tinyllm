from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator
from typing import List, Dict, Optional, Callable

from tinyllm.vector_store import VectorStore
from sentence_transformers import util


class InitValidator(Validator):
    collection_name: str

class InputValidator(Validator):
    user_question: str
    k: Optional[int] = 1
    metadata_filters: Optional[Dict] = {}

class OutputValidator(Validator):
    best_examples: List[Dict]

class VectorStoreExampleSelector(Function):
    def __init__(self, collection_name: str, **kwargs):
        val = InitValidator(collection_name=collection_name)
        super().__init__(
            input_validator=InputValidator,
            output_validator=OutputValidator,
            **kwargs
        )
        self.vector_store = VectorStore()
        self.collection_name = collection_name

    async def run(self, **kwargs):
        results = await self.vector_store.similarity_search(
            query=kwargs['user_question'],
            k=kwargs['k'],
            collection_filters=[self.collection_name]
        )
        return {'best_examples': results}


class LocalExampleInitValidator(Validator):
    examples: List
    embedding_function: Callable


class LocalExampleSelector(Function):
    def __init__(self,
                 examples,
                 embedding_function,
                 **kwargs):
        val = LocalExampleInitValidator(examples=examples, embedding_function=embedding_function)
        super().__init__(
            input_validator=InputValidator,
            output_validator=OutputValidator,
            **kwargs
        )
        self.examples = examples
        self.embedding_function = embedding_function
        for example in examples:
            if example.get('embeddings') is None:
                example['embeddings'] = self.embedding_function(example)
        self.embeddings = [example['embeddings'] for example in examples]

    async def run(self, **kwargs):
        query_embedding = self.embedding_function(kwargs['user_question'])
        max_cos_ind = 0
        for i, embedding in enumerate(self.embeddings):
            if util.cos_sim(query_embedding, embedding)[0] > util.cos_sim(query_embedding, self.embeddings[max_cos_ind])[0]:
                max_cos_ind = i
        return {'best_examples': [self.examples[max_cos_ind]]}