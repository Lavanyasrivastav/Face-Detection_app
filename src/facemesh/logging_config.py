"""Centralized logging setup.

Production code should never rely on ``print`` for diagnostics -- it
can't be filtered, redirected, or leveled. This module gives every
other module a consistently formatted logger.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Safe to call repeatedly (e.g. once per module) -- handlers are only
    attached once per logger name, so log lines are never duplicated.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
