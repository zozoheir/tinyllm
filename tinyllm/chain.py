import abc
import logging
from typing import Any

from tinyllm.service import LLMService
from tinyllm.prompt import Prompt

logger = logging.getLogger(__name__)


class Chain(abc.ABC):

    def __init__(self,
                 llm_service: LLMService,
                 prompt: Prompt = None):
        super().__init__()
        self.llm_service = llm_service
        self.current_model_name = None
        self.current_model_params = None
        self.prompt = prompt

    @abc.abstractmethod
    def parse_output(self, output: Any) -> Any:
        return output

    @abc.abstractmethod
    def process_output(self, output: Any) -> Any:
        return output

    def load(self,
             provider_name: str,
             model_name: str,
             model_params: dict = {}):
        self.current_provider = provider_name
        self.llm_service.providers[provider_name].load(model_name=model_name,
                                                       model_params=model_params)

    def validate_model(self,
                       provider_name,
                       model_name: str) -> Exception:
        if provider_name not in self.llm_service.providers.keys():
            raise ValueError(f"No provider found with name: {provider_name}")
        else:
            if model_name not in self.llm_service.providers[provider_name].llms.keys():
                raise ValueError(f"No model for provider '{provider_name}' and model '{model_name}'")

    def get_output(self,
                   **kwargs):
        self.prompt.validate_inputs(**kwargs)
        model_input = self.prompt.get()
        current_model = self.llm_service.providers[self.current_provider]
        result = current_model.call(model_input=model_input)
        if self.validate_output(result):
            self.process_output(self.parse_output(result))
        else:
            logger.warning("Output validation failed.")
        return result

    def run_chain(self,
                  **kwargs
                  ):
        output = self.get_output(**kwargs)
        self.validate_output(output)
        parsed_output = self.parse_output(output)
        return self.process_output(parsed_output)
