from env_util.environment import openagents_env
from tinyllm.config import App
from tinyllm.logger import get_logger

APP_CONFIG = App()
APP_CONFIG.set_logging('default', get_logger(name='default'))

APP_CONFIG.set_provider('openai', {
    "api_key":
        openagents_env.configs.OPENAI_API_KEY
})
