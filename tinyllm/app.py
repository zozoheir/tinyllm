import os
from typing import Dict

from py2neo import Graph

from tinyllm.logger import get_logger


class App:
    def __init__(self):
        self.logging = {
            'default':get_logger(name='default')
        }
        self.graph_db = None

        try:
            self.connect_graph_db(host=os.environ['TINYLLM_DB_HOST'],
                                        port=os.environ['TINYLLM_DB_PORT'],
                                        user=os.environ['TINYLLM_DB_USER'],
                                        password=os.environ['TINYLLM_DB_PASSWORD'])
            self.check_graph_db_connection()
        except Exception as e:
            print(e)

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


    def check_graph_db_connection(self):
        try:
            result = self.graph_db.run("MATCH (n) RETURN COUNT(n) AS count")
            count = result.evaluate()
            if count is not None:
                print("Graph DB connected")
        except Exception as e:
            print("Graph database connection error:", e)


APP = App()