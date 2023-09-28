from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator
from typing import List, Dict, Optional

from tinyllm.vector_store import VectorStore


class InitValidator(Validator):
    collection_name: str

class InputValidator(Validator):
    user_question: str
    k: Optional[int] = 1
    metadata_filters: Optional[Dict] = {}

class OutputValidator(Validator):
    examples: List[Dict]

class ExampleSelector(Function):
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
        return {'examples': results}

