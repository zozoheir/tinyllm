from tinyllm.config import AppConfig
from tinyllm.logger import get_logger

APP_CONFIG = AppConfig()
APP_CONFIG.set_logging('default', get_logger(name='default'))