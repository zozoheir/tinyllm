from typing import List, Dict

from env_util.environment import openagents_env
from tinyllm.llm_provider import LLMProvider, OpenAIProvider


class LLMService:
    def __init__(self,
                 providers: Dict[str, LLMProvider] = None):
        self.providers = providers

    def add_provider(self,
                     model: LLMProvider):
        self.providers[model.name] = model

    def get_llm(self,
                provider_name,
                model_name: str) -> LLMProvider:
        return self.providers[provider_name].llms[model_name]

    def call_llm(self,
                 provider_name,
                 model_name: str,
                 model_input: str,
                 model_params: Dict[str, str] = {},
                 ) -> str:
        return self.providers[provider_name].call(model_name,
                                                  model_input,
                                                  model_params)



