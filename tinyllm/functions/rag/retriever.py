from typing import Type, List

from tinyllm.function import Function
from tinyllm.functions.rag.context_builder import ContextBuilder
from tinyllm.validator import Validator


class RetrieverInitValidator(Validator):
    context_builder: Type[ContextBuilder]

class RetrieverOutputValidator(Validator):
    context: str

class Retriever(Function):
    def __init__(self,
                 context_builder=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.context_builder = context_builder

    async def run(self, **kwargs):
        retrieved_docs = await self.search(**kwargs)
        context = await self.build_context(retrieved_docs=retrieved_docs,
                                           **kwargs)
        return {"context": context}

    async def search(self, **kwargs):
        docs = []
        return docs

    async def build_context(self, **kwargs):
        final_context = self.context_builder.get_context(
            docs=kwargs['retrieved_docs'],
        )
        return final_context