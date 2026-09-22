# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for logging utilities."""

import io
import logging
import re
import tempfile
import time
from pathlib import Path

import pytest

from isaaclab.utils.logger import ColoredFormatter, RateLimitFilter, configure_logging

pytestmark = pytest.mark.unit

ANSI_PATTERN = re.compile(r"^\033\[[\d;]+m$")


def make_record(level: int, msg: str, args: tuple = ()) -> logging.LogRecord:
    return logging.LogRecord(name="test", level=level, pathname="test.py", lineno=1, msg=msg, args=args, exc_info=None)


@pytest.fixture
def root_logger():
    """Hand out the root logger and restore its handlers and level afterwards."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield root
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_colored_formatter_wraps_message_in_level_color(level):
    formatter = ColoredFormatter("%(name)s - %(levelname)s - %(message)s")
    color = ColoredFormatter.COLORS[level]
    assert ANSI_PATTERN.match(color) and ANSI_PATTERN.match(ColoredFormatter.RESET)

    formatted = formatter.format(make_record(getattr(logging, level), "Test message"))
    assert formatted == f"{color}test - {level} - Test message{ColoredFormatter.RESET}"


def test_rate_limit_filter_blocks_repeated_warnings(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)
    rate_filter = RateLimitFilter(interval_seconds=2)
    assert RateLimitFilter().interval == 5

    # only warnings are rate limited
    for level in (logging.DEBUG, logging.INFO, logging.ERROR):
        assert rate_filter.filter(make_record(level, "message")) is True
        assert rate_filter.filter(make_record(level, "message")) is True

    # duplicates are keyed on the rendered message; distinct messages pass independently
    assert rate_filter.filter(make_record(logging.WARNING, "value=%d", (42,))) is True
    assert rate_filter.filter(make_record(logging.WARNING, "value=%d", (42,))) is False
    assert rate_filter.filter(make_record(logging.WARNING, "value=%d", (99,))) is True
    assert set(rate_filter.last_emitted) == {"value=42", "value=99"}

    # the same warning passes again once the interval has elapsed
    now += 2.5
    assert rate_filter.filter(make_record(logging.WARNING, "value=%d", (42,))) is True
    assert rate_filter.last_emitted["value=42"] == now


def test_configure_logging_stream_only(root_logger):
    root_logger.addHandler(logging.StreamHandler())

    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        logger = configure_logging(logging_level=level, save_logs_to_file=False)
        assert logger is root_logger
        assert logger.level == getattr(logging, level)
        assert len(logger.handlers) == 1  # previous handlers are removed on every call
        assert logger.handlers[0].level == getattr(logging, level)

    (handler,) = configure_logging(logging_level="INFO", save_logs_to_file=False).handlers
    assert isinstance(handler, logging.StreamHandler)
    assert isinstance(handler.formatter, ColoredFormatter)
    assert "%(asctime)s" in handler.formatter._fmt and "%(filename)s" in handler.formatter._fmt
    assert isinstance(handler.filters[0], RateLimitFilter) and handler.filters[0].interval == 5

    captured = io.StringIO()
    handler.setStream(captured)
    module_logger = logging.getLogger("test_module")
    module_logger.info("Test info message")
    module_logger.warning("Test warning message")
    module_logger.debug("Test debug message")
    output = captured.getvalue()
    assert "INFO: Test info message" in output
    assert "WARNING: Test warning message" in output
    assert "Test debug message" not in output


@pytest.mark.parametrize("custom_dir", [True, False])
def test_configure_logging_to_file(root_logger, tmp_path, custom_dir):
    log_dir = tmp_path / "custom_logs" if custom_dir else None
    logger = configure_logging(logging_level="INFO", save_logs_to_file=True, log_dir=log_dir)
    assert logger.level == logging.INFO

    stream_handler, file_handler = logger.handlers
    assert isinstance(stream_handler, logging.StreamHandler)
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.level == logging.DEBUG  # the file keeps everything regardless of the console level
    assert "%(lineno)d" in file_handler.formatter._fmt

    log_file = Path(file_handler.baseFilename)
    expected_dir = log_dir if custom_dir else Path(tempfile.gettempdir()) / "isaaclab" / "logs"
    assert log_file.parent == expected_dir
    assert log_file.is_file()
    assert re.fullmatch(r"isaaclab_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log", log_file.name)
    if not custom_dir:
        file_handler.close()
        log_file.unlink()
