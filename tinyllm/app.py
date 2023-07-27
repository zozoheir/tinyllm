from typing import Dict

from py2neo import Graph

from tinyllm.logger import get_logger


class App:
    def __init__(self):
        self.logging = {
        }
        self.graph_db = None

    def set_logging(self, function_name: str, logger):
        """
        Set logging for a given Function name, or set the 'default' logger.
        :param function_name: Function name
        :param logger: logger

        """
        self.logging[function_name] = logger

    def set_provider(self, name, config: Dict):
        self.providers[name] = config

    def connect_graph_db(self,
                         host,
                         port,
                         user,
                         password):
        self.graph_db = Graph(f"neo4j+s://{host}:{port}", auth=(user, password))


APP_CONFIG = App()
APP_CONFIG.set_logging('default', get_logger(name='default'))
