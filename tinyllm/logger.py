import logging
from logging import StreamHandler

def get_logger(name: str,
               handlers=[StreamHandler()],
               format='%(asctime)s - %(levelname)s - %(message)s',
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
