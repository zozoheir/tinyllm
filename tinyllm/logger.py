import logging
from logging import StreamHandler

def get_logger(name: str,
               handlers=[StreamHandler()],
               format='%(levelname)s - %(asctime)s - msg: %(message)s',
               level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    root_logger = logging.getLogger()
    root_logger.handlers = []
    logger.handlers = []
    for handler in handlers:
        handler.setFormatter(logging.Formatter(format))
        logger.addHandler(handler)
        logger.info("Logger initialized")
    return logger
