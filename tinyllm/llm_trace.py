"""
Wrapper around the Langfuse API to make it easier to use.
"""
import os

from langfuse import Langfuse

from langfuse.model import CreateGeneration, CreateTrace, CreateSpan, UpdateGeneration, UpdateSpan, CreateScore

langfuse_client = Langfuse(
    public_key=os.environ['LANGFUSE_PUBLIC_KEY'],
    secret_key=os.environ['LANGFUSE_SECRET_KEY'],
    host="https://cloud.langfuse.com/"
)

class LLMTrace:

    def __init__(self,
                 **kwargs):
        self.trace = langfuse_client.trace(CreateTrace(**kwargs))

    def create_generation(self,
                          **kwargs):
        self.current_generation = self.trace.generation(CreateGeneration(
            **kwargs))

    def update_generation(self,
                          **kwargs):
        self.current_generation.update(UpdateGeneration(
            **kwargs
        ))

    def create_span(self,
                    **kwargs):
        self.current_span = self.trace.span(CreateSpan(**kwargs))

    def update_span(self,
                    **kwargs):
        self.current_span.update(UpdateSpan(**kwargs))

    def score_generation(self,
                         **kwargs):
        self.current_generation.score(CreateScore(**kwargs))
