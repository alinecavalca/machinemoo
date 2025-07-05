"""
logging_config.py

Configures a centralized logger for the machinemoo project.

- Sets up a logger named "machinemoo" with DEBUG level.
- Adds a StreamHandler with a custom formatter for timestamped logs.
- Suppresses verbose logging from external libraries such as sklearn, pyomo, and gurobipy by setting their log levels to WARNING.
"""

import logging

LOG_NAME = "machinemoo"

logger = logging.getLogger(LOG_NAME)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Silences noisy external libraries
for noisy_lib in ["sklearn", "pyomo", "gurobipy"]:
    logging.getLogger(noisy_lib).setLevel(logging.WARNING)