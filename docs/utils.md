# utils

The `utils` module contains general-purpose utilities to support the machinemoo project. 

## Contents

- **logging_config.py**: Centralized logging configuration with a standardized format and suppression of verbose logs from common external libraries.

## Usage

Import the logger from the module to use consistent logging throughout the project:

```python
from machinemoo.utils.logging_config import get_logger

logger.debug("This is a debug message")
logger.info("Informational message")
```

Feel free to add more utility functions or configurations here as the project grows.