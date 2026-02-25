# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with logging utilities.

To use the logger, you can use the :func:`logging.getLogger` function.

Example:
    >>> import logging
    >>>
    >>> # define logger for the current module (enables fine-control)
    >>> logger = logging.getLogger(__name__)
    >>>
    >>> # log messages
    >>> logger.info("This is an info message")
    >>> logger.warning("This is a warning message")
    >>> logger.error("This is an error message")
    >>> logger.critical("This is a critical message")
    >>> logger.debug("This is a debug message")
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from typing import Literal


def configure_logging(
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
    save_logs_to_file: bool = True,
    log_dir: str | None = None,
) -> logging.Logger:
    """Setup the logger with a colored formatter and a rate limit filter.

    This function defines the default logger for IsaacLab. It adds a stream handler with a colored formatter
    and a rate limit filter. If :attr:`save_logs_to_file` is True, it also adds a file handler to save the logs
    to a file. The log directory can be specified using :attr:`log_dir`. If not provided, the logs will be saved
    to the temp directory with the sub-directory "isaaclab/logs".

    The log file name is formatted as "isaaclab_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log".
    The log record format is "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s".
    The date format is "%Y-%m-%d %H:%M:%S".

    Args:
        logging_level: The logging level.
        save_logs_to_file: Whether to save the logs to a file.
        log_dir: The directory to save the logs to. Default is None, in which case the logs
            will be saved to the temp directory with the sub-directory "isaaclab/logs".

    Returns:
        The root logger.
    """
    root_logger = logging.getLogger()
    # the root logger must be the lowest level to ensure that all messages are logged
    root_logger.setLevel(logging.DEBUG)

    # remove existing handlers
    # Note: iterate over a copy [:] to avoid modifying list during iteration
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # add a stream handler with default level
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)

    # add a colored formatter
    formatter = ColoredFormatter(fmt="%(asctime)s [%(filename)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    handler.addFilter(RateLimitFilter(interval_seconds=5))
    root_logger.addHandler(handler)

    # add a file handler
    if save_logs_to_file:
        # if log_dir is not provided, use the temp directory
        if log_dir is None:
            log_dir = os.path.join(tempfile.gettempdir(), "isaaclab", "logs")
        # create the log directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)
        # create the log file path
        log_file_path = os.path.join(log_dir, f"isaaclab_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

        # create the file handler
        file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # print the log file path once at startup with nice formatting
        cyan = "\033[36m"  # cyan color
        bold = "\033[1m"  # bold text
        reset = "\033[0m"  # reset formatting
        message = f"[INFO][IsaacLab]: Logging to file: {log_file_path}"
        border = "=" * len(message)
        print(f"\n{cyan}{border}{reset}")
        print(f"{cyan}{bold}{message}{reset}")
        print(f"{cyan}{border}{reset}\n")

    # return the root logger
    return root_logger


class ColoredFormatter(logging.Formatter):
    """Colored formatter for logging.

    This formatter colors the log messages based on the log level.
    """

    COLORS = {
        "WARNING": "\033[33m",  # orange/yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[1;31m",  # bold red
        "INFO": "\033[0m",  # reset
        "DEBUG": "\033[0m",
    }
    """Colors for different log levels."""

    RESET = "\033[0m"
    """Reset color."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record.

        Args:
            record: The log record to format.

        Returns:
            The formatted log record.
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


class RateLimitFilter(logging.Filter):
    """Custom rate-limited warning filter.

    This filter allows warning-level messages only once every few seconds per message.
    This is useful to avoid flooding the log with the same message multiple times.
    """

    def __init__(self, interval_seconds: int = 5):
        """Initialize the rate limit filter.

        Args:
            interval_seconds: The interval in seconds to limit the warnings.
                Defaults to 5 seconds.
        """
        super().__init__()
        self.interval = interval_seconds
        self.last_emitted = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Allow warning-level messages only once every few seconds per message.

        Args:
            record: The log record to filter.

        Returns:
            True if the message should be logged, False otherwise.
        """
        # only filter warning-level messages
        if record.levelno != logging.WARNING:
            return True
        # check if the message has been logged in the last interval
        now = time.time()
        msg_key = record.getMessage()
        if msg_key not in self.last_emitted or (now - self.last_emitted[msg_key]) > self.interval:
            # if the message has not been logged in the last interval, log it
            self.last_emitted[msg_key] = now
            return True
        # if the message has been logged in the last interval, do not log it
        return False
