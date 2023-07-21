from typing import Dict, Any

from tinyllm.function import Function

class Service(Function):
    def __init__(self, name, config: Dict[str, Any] = {}):
        self.config = config
        super().__init__(name=name, type='service')
