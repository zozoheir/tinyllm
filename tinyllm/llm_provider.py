import abc
from typing import Dict, Any, List

from langchain import OpenAI


class LLMProvider(abc.ABC):
    def __init__(self,
                 name: str):
        self.name = name
        self.llms = {}

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
                 openai_api_key: str,
                 model_names: List[str],
                 default_params: Dict[str, Any]):
        super().__init__(name='openai')
        self.openai_api_key = openai_api_key
        for model_name in model_names:
            self.llms[model_name] = OpenAI(model_name=model_name,
                                           openai_api_key=openai_api_key,
                                           **default_params)

    def call(self,
             model_name,
             model_input: str,
             model_params: Dict[str, Any] = {}) -> Any:
        return self.llms[model_name].predict(model_input,
                                             **model_params)

    def parse_call(self, result: Any) -> Dict[str, Any]:
        pass
