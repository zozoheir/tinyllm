import logging
import os
from pathlib import Path
import yaml


def search_and_load_yaml(yaml_file_name, directories):
    for directory in directories:
        yaml_path = directory / yaml_file_name
        if yaml_path.is_file():
            print(f"CONFIG: tinyllm config found at {yaml_path}")
            with open(yaml_path, 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                    for key, value in config.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                os.environ[subkey] = str(subvalue)
                        else:
                            os.environ[key] = str(value)
                except yaml.YAMLError as exc:
                    print(exc)
                break


directories = [
    Path.cwd(),
    Path.home(),
    Path.home() / 'Documents'
]
search_and_load_yaml('tinyllm.yaml', directories)
