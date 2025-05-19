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