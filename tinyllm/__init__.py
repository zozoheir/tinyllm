import yaml
import os

from pathlib import Path, PosixPath


from smartpy.utility.log_util import getLogger
from langfuse import Langfuse

logger = getLogger(__name__)


def load_yaml_config(yaml_file_name: str, directories: list) -> dict:
    config = None
    for directory in directories:
        yaml_path = Path(directory) / yaml_file_name
        if yaml_path.is_file():
            logger.info(f"Tinyllm: config found at {yaml_path}")
            with open(yaml_path, 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    logger.info(exc)
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
    Path.home() / 'Documents',
]

directories.append(PosixPath(os.environ['TINYLLM_YAML_DIR'].replace(' ', '')))
tinyllm_config = load_yaml_config('tinyllm.yaml', directories)

os.environ['OPENAI_API_KEY'] = tinyllm_config['LLM_PROVIDERS']['OPENAI_API_KEY']

langfuse_client = Langfuse(
    public_key=tinyllm_config['LANGFUSE']['PUBLIC_KEY'],
    secret_key=tinyllm_config['LANGFUSE']['SECRET_KEY'],
    host=tinyllm_config['LANGFUSE']['HOST'],
    debug=False,
    flush_interval=0.1,
)

