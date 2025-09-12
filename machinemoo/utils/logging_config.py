"""
logging_config.py

Configures a centralized logger for the machinemoo project.

- Sets up a logger named "machinemoo" with DEBUG level.
- Adds a StreamHandler with a custom formatter for timestamped logs.
- Suppresses verbose logging from external libraries such as sklearn, pyomo, and gurobipy by setting their log levels to WARNING.
"""

import logging

LOG_NAME = "machinemoo"


def get_logger(name: str = LOG_NAME, verbose: bool=False, debug=False) -> logging.Logger:
    """Create and return a logger with a standardized configuration.

    Args:
        name (str): Logger name. Default = 'machinemoo.
        verbose (bool): Enable INFO logging (ignored if `debug` is True). Default: False.
        debug (bool): Enable DEBUG logging, taking priority over `verbose`. Default: False. 
                      If both are False, the logger defaults to ERROR level.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoids root logger interference

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Silence noisy external libraries globally
    for noisy_lib in ["sklearn", "pyomo", "gurobipy"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    return logger

