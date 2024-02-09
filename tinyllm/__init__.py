import yaml
import os

from pathlib import Path, PosixPath

from smartpy.utility import os_util
from smartpy.utility.log_util import getLogger
from langfuse import Langfuse

logger = getLogger(__name__)

global tinyllm_config
global langfuse_client

tinyllm_config = None
langfuse_client = None

def load_yaml_config(yaml_file_path: str) -> dict:
    with open(yaml_file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML file: {exc}")
            raise exc


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
            logger.info(f"Tinyllm: config found at {yaml_path}")
            return yaml_path


# Directories to look for the config file, in order of priority
directories = [
    Path.cwd() if Path.cwd().name != 'tinyllm' else None,
    Path.home(),
    Path.home() / 'Documents',
]

if langfuse_client is None and tinyllm_config is None:
    env_variable_path = os.environ.get('TINYLLM_CONFIG_PATH', '').strip()
    if os_util.isFilePath(env_variable_path):
        logger.info(f"Tinyllm: using config from env variable TINYLLM_CONFIG_PATH: {env_variable_path}")
        set_config(env_variable_path)
    else:
        logger.info(f"Tinyllm: looking for config in directories")
        found_config_path = find_yaml_config('tinyllm.yaml', directories)
        if found_config_path is None:
            raise FileNotFoundError(f"Please provide a config file for tinyllm")
        set_config(found_config_path)