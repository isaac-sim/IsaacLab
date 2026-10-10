# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for logging utilities."""

import logging
import os
import re
import tempfile
from types import SimpleNamespace

import pytest

import isaaclab.utils.logger as logger_module
from isaaclab.utils.logger import ColoredFormatter, RateLimitFilter, configure_logging

pytestmark = pytest.mark.unit


# Fixtures
@pytest.fixture
def test_message():
    """Fixture providing a test message string."""
    return "Test message"


@pytest.fixture
def rate_limit_filter():
    """Fixture providing a RateLimitFilter instance with 2 second interval."""
    return RateLimitFilter(interval_seconds=2)


"""
Tests for the ColoredFormatter class.
"""


@pytest.mark.parametrize(
    "level, color",
    [
        (logging.DEBUG, "\033[0m"),
        (logging.INFO, "\033[0m"),
        (logging.WARNING, "\033[33m"),
        (logging.ERROR, "\033[31m"),
        (logging.CRITICAL, "\033[1;31m"),
    ],
)
def test_level_formatting(test_message, level, color):
    """Test that each level is wrapped in its ANSI color and the reset code, using a custom format string."""
    custom_formatter = ColoredFormatter("%(name)s - %(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="custom.logger",
        level=level,
        pathname="test.py",
        lineno=1,
        msg=test_message,
        args=(),
        exc_info=None,
    )
    formatted = custom_formatter.format(record)

    # DEBUG and INFO use the reset code (no color); every message ends with the reset code
    assert formatted == f"{color}custom.logger - {logging.getLevelName(level)} - {test_message}\033[0m"


"""
Tests for the RateLimitFilter class.
"""


def test_non_warning_messages_pass_through(rate_limit_filter):
    """Test that non-WARNING messages always pass through the filter."""
    # Test INFO
    info_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Info message",
        args=(),
        exc_info=None,
    )
    assert rate_limit_filter.filter(info_record) is True

    # Test ERROR
    error_record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="Error message",
        args=(),
        exc_info=None,
    )
    assert rate_limit_filter.filter(error_record) is True

    # Test DEBUG
    debug_record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg="Debug message",
        args=(),
        exc_info=None,
    )
    assert rate_limit_filter.filter(debug_record) is True

    # the default interval is 5 seconds
    assert RateLimitFilter().interval == 5


def test_warning_after_interval_passes(monkeypatch):
    """Test that duplicate WARNING messages are blocked within the interval and pass after it."""
    now = [100.0]
    monkeypatch.setattr(logger_module, "time", SimpleNamespace(time=lambda: now[0]))
    message = "Rate limited warning"
    filter_short = RateLimitFilter(interval_seconds=1)

    # First warning should pass
    record1 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    assert filter_short.filter(record1) is True

    # Immediate duplicate should be blocked
    record2 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=2,
        msg=message,
        args=(),
        exc_info=None,
    )
    assert filter_short.filter(record2) is False

    # Advance the clock past the interval
    now[0] += 1.1

    # After interval, same message should pass again
    record3 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=3,
        msg=message,
        args=(),
        exc_info=None,
    )
    assert filter_short.filter(record3) is True


def test_formatted_message_warnings(rate_limit_filter):
    """Test rate limiting with formatted WARNING messages."""
    # Test with string formatting
    record1 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="Warning: value=%d",
        args=(42,),
        exc_info=None,
    )
    assert rate_limit_filter.filter(record1) is True

    # Same formatted message should be blocked
    record2 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=2,
        msg="Warning: value=%d",
        args=(42,),
        exc_info=None,
    )
    assert rate_limit_filter.filter(record2) is False

    # Different args create different message, should pass
    record3 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=3,
        msg="Warning: value=%d",
        args=(99,),
        exc_info=None,
    )
    assert rate_limit_filter.filter(record3) is True


"""
Tests for the configure_logging function.
"""


def test_configure_logging_basic():
    """Test basic configure_logging functionality without file logging."""
    # Setup logger without file logging
    logger = configure_logging(logging_level="INFO", save_logs_to_file=False)

    # Should return root logger
    assert logger is not None
    assert logger is logging.getLogger()
    # Root logger level matches the requested level
    assert logger.level == logging.INFO

    # Should have exactly one handler (stream handler)
    assert len(logger.handlers) == 1

    # Stream handler should have ColoredFormatter
    stream_handler = logger.handlers[0]
    assert isinstance(stream_handler, logging.StreamHandler)
    assert isinstance(stream_handler.formatter, ColoredFormatter)
    assert stream_handler.level == logging.INFO

    # Should have RateLimitFilter
    assert len(stream_handler.filters) > 0
    rate_filter = stream_handler.filters[0]
    assert isinstance(rate_filter, RateLimitFilter)
    assert rate_filter.interval == 5


def test_configure_logging_with_file():
    """Test configure_logging with file logging enabled in a custom (nested) log directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_log_dir = os.path.join(temp_dir, "custom_logs")
        logger = configure_logging(logging_level="DEBUG", save_logs_to_file=True, log_dir=custom_log_dir)

        # Should return root logger
        assert logger is not None
        # Root logger level matches the requested level
        assert logger.level == logging.DEBUG

        # Should have two handlers (stream + file)
        assert len(logger.handlers) == 2

        # Check stream handler
        stream_handler = logger.handlers[0]
        assert isinstance(stream_handler, logging.StreamHandler)
        assert isinstance(stream_handler.formatter, ColoredFormatter)
        assert stream_handler.level == logging.DEBUG
        stream_format = stream_handler.formatter._fmt  # type: ignore
        assert "%(asctime)s" in stream_format
        assert "%(filename)s" in stream_format

        # Check file handler: always DEBUG, with a more detailed format including line numbers
        file_handler = logger.handlers[1]
        assert isinstance(file_handler, logging.FileHandler)
        assert file_handler.level == logging.DEBUG
        file_format = file_handler.formatter._fmt  # type: ignore
        assert "%(asctime)s" in file_format
        assert "%(lineno)d" in file_format

        # Custom directory should be created and hold the log file
        assert os.path.isdir(custom_log_dir)
        assert os.path.dirname(file_handler.baseFilename) == custom_log_dir
        log_files = [f for f in os.listdir(custom_log_dir) if f.startswith("isaaclab_")]
        assert len(log_files) == 1

        # Check filename format: isaaclab_YYYY-MM-DD_HH-MM-SS.log
        pattern = r"isaaclab_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log"
        assert re.match(pattern, log_files[0]), f"Log filename {log_files[0]} doesn't match expected pattern"


def test_configure_logging_levels():
    """Test configure_logging with different logging levels."""
    from typing import Literal

    levels: list[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ]
    level_values = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    for level_str in levels:
        logger = configure_logging(logging_level=level_str, save_logs_to_file=False)
        # Root logger level matches the requested level
        assert logger.level == level_values[level_str]
        # Handler level should match the requested level
        assert logger.handlers[0].level == level_values[level_str]


def test_configure_logging_removes_existing_handlers():
    """Test that configure_logging removes existing handlers."""
    # Get root logger and add a dummy handler
    root_logger = logging.getLogger()
    dummy_handler = logging.StreamHandler()
    root_logger.addHandler(dummy_handler)

    initial_handler_count = len(root_logger.handlers)
    assert initial_handler_count > 0

    # Setup logger should remove existing handlers
    logger = configure_logging(logging_level="INFO", save_logs_to_file=False)

    # Should only have the new handler
    assert len(logger.handlers) == 1
    assert dummy_handler not in logger.handlers


def test_configure_logging_default_log_dir():
    """Test configure_logging uses temp directory when log_dir is None."""

    logger = configure_logging(logging_level="INFO", save_logs_to_file=True, log_dir=None)

    # Root logger level matches the requested level
    assert logger.level == logging.INFO

    # Should have file handler
    assert len(logger.handlers) == 2
    file_handler = logger.handlers[1]
    assert isinstance(file_handler, logging.FileHandler)
    # The file handler always records DEBUG, regardless of the requested console level.
    assert file_handler.level == logging.DEBUG

    # File should be in temp directory
    log_file_path = file_handler.baseFilename
    assert os.path.dirname(log_file_path) == os.path.join(tempfile.gettempdir(), "isaaclab", "logs")
    assert os.path.basename(log_file_path).startswith("isaaclab_")

    # Cleanup
    if os.path.exists(log_file_path):
        os.remove(log_file_path)


def test_configure_logging_actual_logging():
    """Test that logger actually logs messages correctly."""
    import io

    # Capture stdout
    captured_output = io.StringIO()

    # Setup logger
    logger = configure_logging(logging_level="INFO", save_logs_to_file=False)

    # Temporarily redirect handler to captured output
    stream_handler = logger.handlers[0]
    assert isinstance(stream_handler, logging.StreamHandler)
    original_stream = stream_handler.stream  # type: ignore
    stream_handler.stream = captured_output  # type: ignore

    # Log some messages
    test_logger = logging.getLogger("test_module")
    test_logger.info("Test info message")
    test_logger.warning("Test warning message")
    test_logger.debug("Test debug message")  # Should not appear (level is INFO)

    # Restore original stream
    stream_handler.stream = original_stream  # type: ignore

    # Check output
    output = captured_output.getvalue()
    assert "Test info message" in output
    assert "Test warning message" in output
    assert "Test debug message" not in output  # DEBUG < INFO
    assert "INFO" in output
    assert "WARNING" in output
