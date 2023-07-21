import logging
from logging import StreamHandler

from tinyllm.validator import Validator


class LoggerValidator(Validator):
    name: str
    level: int = logging.INFO
    handlers: list = None
    formatter: logging.Formatter = None


def get_logger(name: str,
               level: int = logging.INFO,
               handlers=None, formatter=None):
    logger = logging.getLogger()
    logger.setLevel(level)

    if handlers is None:
        handlers = [StreamHandler()]
    for handler in handlers:
        if formatter is not None:
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


default_logger = get_logger(name=__name__,
                            level=logging.INFO,
                            handlers=[StreamHandler()])
