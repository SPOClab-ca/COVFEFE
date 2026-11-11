import logging
import os

# Worker processes started with the "spawn" start method (the default on macOS since
# Python 3.8 and on Linux since 3.14) re-import this module with a fresh, unset logger.
# The chosen log file is therefore passed to children through an environment variable
# and the logger is created lazily on first use.
_LOG_FILE_ENV = "COVFEFE_LOG_FILE"

logger = None


def set_logger(filename=None):
    global logger

    if filename:
        os.environ[_LOG_FILE_ENV] = filename
        logging.basicConfig(filename=filename, level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    if logger is None:
        set_logger(os.environ.get(_LOG_FILE_ENV))
    return logger
