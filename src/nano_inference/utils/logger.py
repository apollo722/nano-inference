import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "nano-inference",
    level: int = logging.INFO,
    log_format: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    logger.setLevel(level)

    if log_format is None:
        log_format = (
            "%(asctime)s.%(msecs)03d [%(levelname)s] "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    formatter = logging.Formatter(
        fmt=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = setup_logger()
