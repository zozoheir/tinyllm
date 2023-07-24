from typing import Dict

from tinyllm.logger import get_logger

class App:
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


    def set_logging(self, function_name: str, logger):
        """
        Set logging for a given Function name, or set the 'default' logger.
        :param function_name: Function name
        :param logger: logger

        """
        self.logging[function_name] = logger

    def set_provider(self, name, config: Dict):
        self.providers[name] = config


APP_CONFIG = App()
APP_CONFIG.set_logging('default', get_logger(name='default'))