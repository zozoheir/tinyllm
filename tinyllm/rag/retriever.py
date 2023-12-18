from abc import abstractmethod

from tinyllm.function import Function
from tinyllm.rag.document.store import DocumentStore
from tinyllm.validator import Validator


class RetrieverInitValidator(Validator):
    context_builder: DocumentStore


class RetrieverInputValidator(Validator):
    input: str


class RetrieverOutputValidator(Validator):
    context: str


class Retriever(Function):
    def __init__(self,
                 context_builder=None,
                 **kwargs):
        super().__init__(
            input_validator=RetrieverInputValidator,
            output_validator=RetrieverOutputValidator,
            **kwargs)
        self.document_store = context_builder
        self.search_results = {}

    @abstractmethod
    async def search(self, **kwargs):
        pass

    async def run(self, **kwargs):
        retrieved_docs = await self.search(**kwargs)
        context = await self.build_context(retrieved_docs=retrieved_docs,
                                           **kwargs)
        return {"context": context}

    async def build_context(self, **kwargs):
        final_context = self.document_store.format(
            search_results=kwargs['retrieved_docs'],
        )
        return final_context
