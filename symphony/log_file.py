"""Logging configuration with rotating file handler."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def default_log_file(root: Optional[str] = None) -> str:
    """Return the default log file path."""
    base = root or os.getcwd()
    return os.path.join(base, "log", "symphony.log")


def configure_logging(
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    level: int = logging.INFO,
    console: bool = True,
):
    """Configure rotating file logger.

    Args:
        log_file: Path to log file. Defaults to cwd/log/symphony.log.
        max_bytes: Max size per log file (default 10MB).
        backup_count: Number of backup files to keep (default 5).
        level: Logging level.
        console: Whether to also log to console.
    """
    log_file = log_file or default_log_file()
    log_dir = os.path.dirname(log_file)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("symphony")
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)

    return root_logger
