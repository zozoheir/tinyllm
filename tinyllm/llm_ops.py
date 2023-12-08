"""
Wrapper around the Langfuse API to make it easier to use.
"""
import json
import os

import pydantic
from langfuse import Langfuse
import langfuse
from langfuse.api import CreateDatasetItemRequest
from langfuse.model import CreateGeneration, CreateDatasetRequest, CreateTrace, CreateSpan, UpdateGeneration, \
    UpdateSpan, CreateScore

langfuse_client = Langfuse(
    public_key=os.environ['LANGFUSE_PUBLIC_KEY'],
    secret_key=os.environ['LANGFUSE_SECRET_KEY'],
    host="https://us.cloud.langfuse.com/",
    debug=False,
    flush_interval=0.1,
)


class LLMDataset:

    def __init__(self,
                 name):
        self.name = name

        try:
            self.dataset = langfuse_client.get_dataset(name=self.name)
        except pydantic.error_wrappers.ValidationError:
            self.dataset = langfuse_client.create_dataset(CreateDatasetRequest(name=self.name))

    def create_item(self,
                    **kwargs):
        self.current_item = langfuse_client.create_dataset_item(
            CreateDatasetItemRequest(dataset_name=self.name,
                                     **kwargs))
        return self.current_item


class LLMTrace:

    def __init__(self,
                 **kwargs):
        self.trace = langfuse_client.trace(CreateTrace(**kwargs))
        self.current_generation = None
        self.current_span = None

    def create_generation(self,
                          **kwargs):
        self.current_generation = self.trace.generation(CreateGeneration(**kwargs))
        kwargs.keys()

    def update_generation(self,
                          **kwargs):
        # kwargs['completion'] = json.dumps({'completion': kwargs.get('completion', "No completion")})
        self.current_generation.update(UpdateGeneration(
            **kwargs
        ))

    def create_span(self,
                    **kwargs):
        new_span = self.trace.span(CreateSpan(**kwargs))
        self.current_span = new_span
        return new_span

    def update_span(self,
                    span=None,
                    **kwargs):
        if span is None:
            self.current_span.update(UpdateSpan(**kwargs))
        else:
            span.update(UpdateSpan(**kwargs))

    def score_generation(self,
                         **kwargs):
        self.current_generation.score(CreateScore(**kwargs))
