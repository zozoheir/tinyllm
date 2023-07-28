from typing import Dict
import yaml
import os


from tinyllm.logger import get_logger
from pathlib import Path

from py2neo import Graph

def load_yaml_config(yaml_file_name: str, directories: list) -> dict:
    config = None
    for directory in directories:
        yaml_path = Path(directory) / yaml_file_name
        if yaml_path.is_file():
            print(f"CONFIG: tinyllm config found at {yaml_path}")
            with open(yaml_path, 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
                break
    return config


def set_env_variables(config: dict):
    if config is not None:
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    os.environ[subkey] = str(subvalue)
            else:
                os.environ[key] = str(value)


directories = [
    Path.cwd(),
    Path.home(),
    Path.home() / 'Documents'
]


class App:
    def __init__(self):
        self.logging = {
            'default':get_logger(name='default')
        }
        self.graph_db = None
        self.config = load_yaml_config('tinyllm.yaml', directories)
        if self.config is not None:
            set_env_variables(self.config)
            print("CONFIG: tinyllm config loaded")
        else:
            raise Exception("CONFIG: tinyllm config not found")

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
                print("CONFIG: Graph DB connected")
        except Exception as e:
            print("CONFIG: Graph database connection error:", e)


APP = App()
