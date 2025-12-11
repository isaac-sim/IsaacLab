# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with logging utilities."""

from __future__ import annotations

import logging
import time

# import logger
logger = logging.getLogger(__name__)


# --- Colored formatter ---
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "WARNING": "\033[33m",  # orange/yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[31m",  # red
        "INFO": "\033[0m",  # reset
        "DEBUG": "\033[0m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


# --- Custom rate-limited warning filter ---
class RateLimitFilter(logging.Filter):
    def __init__(self, interval_seconds=5):
        super().__init__()
        self.interval = interval_seconds
        self.last_emitted = {}

    def filter(self, record):
        """Allow WARNINGs only once every few seconds per message."""
        if record.levelno != logging.WARNING:
            return True
        now = time.time()
        msg_key = record.getMessage()
        if msg_key not in self.last_emitted or (now - self.last_emitted[msg_key]) > self.interval:
            self.last_emitted[msg_key] = now
            return True
        return False
