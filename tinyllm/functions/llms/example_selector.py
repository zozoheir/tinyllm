from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator
from typing import List, Dict

from tinyllm.vector_store import VectorStore

class ExampleSelectorInputValidator(Validator):
    user_question: str

class ExampleSelectorOutputValidator(Validator):
    examples: List[Dict]

class ExampleSelector(Function):
    def __init__(self, collection_name: str, k: int = 5, **kwargs):
        super().__init__(
            input_validator=ExampleSelectorInputValidator,
            output_validator=ExampleSelectorOutputValidator,
            **kwargs
        )
        self.vector_store = VectorStore()
        self.collection_name = collection_name
        self.k = k

    async def run(self, **kwargs):
        results = await self.vector_store.similarity_search(
            query=kwargs['user_question'],
            k=self.k,
            collection_filters=[self.collection_name]
        )
        return {'examples': results}

