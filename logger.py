import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.addFilter(lambda r: r.levelno == logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt="%H:%M:%S",
    ))

    logger.addHandler(handler)
    logger.propagate = False
    return logger
