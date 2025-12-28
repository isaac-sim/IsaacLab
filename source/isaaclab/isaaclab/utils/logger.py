# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with logging utilities.

To use the logger, you can use the :func:`logging.getLogger` function.

Example:
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("This is an info message")
    >>> logger.warning("This is a warning message")
    >>> logger.error("This is an error message")
    >>> logger.critical("This is a critical message")
    >>> logger.debug("This is a debug message")
"""

from __future__ import annotations

import logging
import time

# import logger
logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for logging.

    This formatter colors the log messages based on the log level.
    """

    COLORS = {
        "WARNING": "\033[33m",  # orange/yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[31m",  # red
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
