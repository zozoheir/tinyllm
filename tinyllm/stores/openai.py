from typing import Any, Dict

from langchain import OpenAI

from tinyllm.functions.function import Function
from tinyllm.services.provider import Store


open_ai_max_context_tokens = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "text-ada-001": 2049,
    "ada": 2049,
    "text-babbage-001": 2040,
    "babbage": 2049,
    "text-curie-001": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}


# Open AI model and provider
class OpenAILLM(Function):
    def __init__(self,
                 llm_name,
                 llm_params,
                 provider_kwargs):
        super().__init__(name=llm_name)
        self.llm_name = llm_name
        self.llm_params = llm_params
        self.llm = OpenAI(llm_name=llm_name, **provider_kwargs, **llm_params)

    def input(self, input):
        # You can also call the openai API directly here and apply whatever io transforms you'd like
        self.llm.predict(input)


class LLMStore(Store):
    def __init__(self, llm_base: OpenAILLM, provider_kwargs: Dict[str,Any]):
        super().__init__(name='openai', llm_base=llm_base, provider_kwargs=provider_kwargs)
        self.provider_kwargs = provider_kwargs
