import os

from langfuse import Langfuse
from langfuse.api.model import CreateTrace, CreateGeneration, UpdateGeneration, CreateSpan, UpdateSpan, CreateScore

langfuse_client = Langfuse(
    public_key=os.environ['LANGFUSE_PUBLIC_KEY'],
    secret_key=os.environ['LANGFUSE_SECRET_KEY'],
    host="https://cloud.langfuse.com/"
)


class LLMTrace:

    def __init__(self,
                 is_traced=True,
                 **kwargs):
        self.is_traced = is_traced
        if self.is_traced is True:
            self.trace = langfuse_client.trace(CreateTrace(**kwargs))

    def create_generation(self,
                          **kwargs):
        if self.is_traced is True:
            self.current_generation = self.trace.generation(CreateGeneration(
                **kwargs))

    def update_generation(self,
                          **kwargs):
        if self.is_traced is True:
            self.current_generation.update(UpdateGeneration(
                **kwargs
            ))

    def create_span(self,
                    **kwargs):
        if self.is_traced is True:
            self.current_span = self.trace.span(CreateSpan(
                **kwargs
            )
            )

    def update_span(self,
                    **kwargs):
        if self.is_traced is True:
            self.current_span.update(UpdateSpan(**kwargs))

    def score_generation(self,
                         **kwargs):
        if self.is_traced is True:
            self.current_generation.score(CreateScore(
                **kwargs
            ))
