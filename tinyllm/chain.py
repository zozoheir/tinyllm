import abc
import logging

from tinyllm.models import LLMService
from tinyllm.prompt import Prompt

logger = logging.getLogger(__name__)


class Chain(abc.ABC):

    def __init__(self,
                 llm_service: LLMService,
                 prompt: Prompt = None):
        super().__init__()
        self.llm_service = llm_service
        self.current_model = None
        self.prompt = prompt

    def validate_model(self,
                       provider_name,
                       model_name: str) -> Exception:
        if provider_name not in self.llm_service.providers.keys():
            raise ValueError(f"No provider found with name: {provider_name}")
        else:
            if model_name not in self.llm_service.providers[provider_name].llms.keys():
                raise ValueError(f"No model for provider '{provider_name}' and model '{model_name}'")

    def set_model(self,
                  provider_name,
                  model_name: str):
        self.validate_model(provider_name, model_name)
        self.current_provider = provider_name
        self.current_model = model_name

    def get_output(self, **kwargs):
        self.prompt.validate_inputs(**kwargs)
        model_input = self.prompt.get()
        result = self.llm_service.providers[self.current_provider].call(model_name=self.current_model,
                                                                        model_input=model_input)
        if self.validate_output(result):
            self.process_output(self.parse_output(result))
        else:
            logger.warning("Output validation failed.")
        return result

    def run_chain(self,
                  **kwargs):
        output = self.get_output(**kwargs)
        self.validate_output(output)
        parsed_output = self.parse_output(output)
        return self.process_output(parsed_output)
