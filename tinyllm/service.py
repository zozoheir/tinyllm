import abc
from typing import List, Dict, Any

from env_util.environment import openagents_env
from tinyllm.provider import LLMProvider, OpenAIProvider


class LLMService:
    def __init__(self,
                 providers: Dict[str, LLMProvider] = None):
        self.providers = providers

    @abc.abstractmethod
    def load(self,
             provider_name: str,
             model_name: str,
             model_params: Dict[str, Any]):
        self.providers[provider_name].load(model_name=model_name,
                                           model_params=model_params)
        pass

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
