from typing import Dict, Any

from tinyllm.stores import LLMProviders
from tinyllm.stores.openai import OpenAILLM
from tinyllm.store import Service


class LLMService(Service):
    def __init__(self,
                 service_config):
        #TODO Change LLMProviders to dict so you can easily iterate
        self.providers = {
            'openai': LLMProviders.OPENAI(model=OpenAILLM,
                                          provider_kwargs=service_config['openai'])
        }
        self.model = None

    def set_model(self, provider_name: str, llm_name: str, llm_params: Dict[str, Any]):
        self.model = self.providers[provider_name].get_model(llm_name=llm_name, llm_params=llm_params)