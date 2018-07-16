import logging

logger = None

def set_logger(filename=None):
    global logger

    if filename:
        logging.basicConfig(filename='log.log',level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    return logger