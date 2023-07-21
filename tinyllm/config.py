from typing import Dict

from tinyllm.logger import get_logger


class AppConfig:
    def __init__(self):
        self.logging = {
        }
        self.providers = {
            'providers': {
                'openai': None,
                'huggingface': None,
                'anthropic': None,
            }
        }

    def set_logging(self, key, value):
        self.logging[key] = value

    def set_provider(self, name, config: Dict):
        self.providers[name] = config


APP_CONFIG = AppConfig()
APP_CONFIG.set_logging('default', get_logger(name='default'))