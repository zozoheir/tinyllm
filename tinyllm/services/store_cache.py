from typing import Dict, Any

from tinyllm.functions.function import Function


class ModelCache:
    def __init__(self):
        self.models = {}
        self.last_model = None

    def add(self,
            model: Function):
        self.models[str((model.llm_name, model.llm_params))] = model

    def get(self,
            llm_name: str,
            llm_params: Dict[str, Any]) -> Any:
        return self.models[str((llm_name, llm_params))]

    @property
    def index(self):
        return [(model.llm_name, model.llm_params) for model in list(self.models.keys())]