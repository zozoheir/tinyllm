from typing import Dict, Any

from tinyllm.operator import Operator

class Service(Operator):
    def __init__(self, name, config: Dict[str, Any] = {}):
        self.config = config
        super().__init__(name=name, type='service')
