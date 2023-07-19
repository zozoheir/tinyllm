import abc
from enum import Enum
from typing import Dict, Any, List

from langchain import OpenAI


class LLMProvider(abc.ABC):
    def __init__(self,
                 name: str):
        self.name = name
        self.llms = {}

    @abc.abstractmethod
    def load(self,
             model_name: str,
             model_params: Dict[str, Any]) -> Any:
        pass

    @abc.abstractmethod
    def parse_call(self, result: Any) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def call(self,
             model_input: str,
             model_params: Dict) -> Any:
        pass


class OpenAIProvider(LLMProvider):

    def __init__(self,
                 openai_api_key: str):
        super().__init__(name='openai')
        self.openai_api_key = openai_api_key

    def load(self,
             model_name: str = None,
             model_params: Dict[str, Any] = {}) -> Any:
        self.llms[model_name] = OpenAI(model_name=model_name,
                                       openai_api_key=self.openai_api_key,
                                       **model_params)
        self.current_model = self.llms[model_name]

    def call(self,
             model_input: str) -> Any:
        if not self.current_model:
            raise ValueError("No model loaded.")
        return self.current_model.predict(model_input)

    def parse_call(self, result: Any) -> Dict[str, Any]:
        pass


class LLMProviders:
    OPENAI = OpenAIProvider