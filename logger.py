"""
src/utils/logger.py
Industry-standard structured logger with file + console handlers.
"""

import logging
import sys
import os
from datetime import datetime

def get_logger(name: str, log_dir: str = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh  = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
