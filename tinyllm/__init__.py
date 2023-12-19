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
    config = None
    yaml_path = Path(yaml_file_path)
    if yaml_path.is_file():
        logger.info(f"Tinyllm: config found at {yaml_path}")
        with open(yaml_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(f"Error loading YAML file: {exc}")
                raise exc
    else:
        logger.error(f"Config file not found at {yaml_path}")
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
        debug=False,
        flush_interval=0.1,
    )


def find_yaml_config(yaml_file_name: str, directories: list) -> dict:
    for directory in directories:
        yaml_path = Path(directory) / yaml_file_name
        if yaml_path.is_file():
            logger.info(f"Tinyllm: config found at {yaml_path}")
            return yaml_path


# Directories to look for the config file, in order of priority
env_variable_path = os.environ.get('TINYLLM_CONFIG_PATH', '').strip()
directories = [
    Path.cwd(),
    Path.home(),
    Path.home() / 'Documents',
]

if langfuse_client is None and tinyllm_config is None:

    if os_util.isFilePath(env_variable_path):
        set_config(env_variable_path)
    else:
        found_config_path = find_yaml_config('tinyllm.yaml', directories)
        set_config(found_config_path)

