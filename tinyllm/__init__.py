import yaml
import os
import pyperclip

from langfuse import Langfuse

import logging
from logging import StreamHandler, Formatter
from pathlib import Path

tinyllm_logger = logging.getLogger('tinyllm')
tinyllm_logger.propagate = False
tinyllm_logger.setLevel(logging.DEBUG)
formatter = Formatter('%(levelname)s | %(name)s | %(asctime)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
tinyllm_logger.addHandler(ch)



global tinyllm_config
global langfuse_client

tinyllm_config = None
langfuse_client = None

def load_yaml_config(yaml_file_path: str) -> dict:
    config = None
    yaml_path = Path(yaml_file_path.strip())
    if yaml_path.is_file():
        with open(yaml_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                tinyllm_logger.error(f"Error loading YAML file: {exc}")
                raise exc
    else:
        tinyllm_logger.error(f"Config file not found at {yaml_path}")
        raise FileNotFoundError(f"No config file at {yaml_path}")
    return config


def set_config(file_path: str):


    global tinyllm_config
    global langfuse_client

    # Load config file

    tinyllm_config = load_yaml_config(file_path)

    # Set LLM providers env vars from the config file
    for provider_key in tinyllm_config['LLM_PROVIDERS'].keys():
        os.environ[provider_key] = tinyllm_config['LLM_PROVIDERS'][provider_key]

    # Initialize Langfuse client
    langfuse_client = Langfuse(
        public_key=tinyllm_config['LANGFUSE']['PUBLIC_KEY'],
        secret_key=tinyllm_config['LANGFUSE']['SECRET_KEY'],
        host=tinyllm_config['LANGFUSE']['HOST'],
        flush_interval=0.1,
    )


def find_yaml_config(yaml_file_name: str, directories: list) -> dict:
    for directory in directories:
        if directory is None:
            continue
        yaml_path = Path(directory) / yaml_file_name
        if yaml_path.is_file():
            tinyllm_logger.info(f"Tinyllm: config found at {yaml_path}")
            return yaml_path


# Directories to look for the config file, in order of priority

directories = [
    Path.cwd() if Path.cwd().name != 'tinyllm' else None,
    Path.home(),
    Path.home() / 'Documents',
]

if langfuse_client is None and tinyllm_config is None:
    tinyllm_config_file_path = os.environ.get('TINYLLM_CONFIG_PATH', None)
    tinyllm_logger.info(f"TINYLLM_CONFIG_PATH: {tinyllm_config_file_path}")
    if tinyllm_config_file_path is not None and tinyllm_config_file_path != '':
        set_config(tinyllm_config_file_path)
    else:
        tinyllm_logger.info(f"Tinyllm: no config file path provided, searching for config file")
        found_config_path = find_yaml_config('tinyllm.yaml', directories)
        if found_config_path is None:
            raise FileNotFoundError(f"Please provide a config file for tinyllm")
        set_config(found_config_path)




def get_agent_code(system_role):
    definition = f"""
tiny_agent = Agent(
            name='My Agent',
            system_role='{system_role}',
            output_model=None
    )
    """
    pyperclip.copy(definition)


