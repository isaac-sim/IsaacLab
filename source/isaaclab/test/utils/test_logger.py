# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for logging utilities."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import logging
import os
import re
import tempfile
import time

import pytest

from isaaclab.utils.logger import ColoredFormatter, RateLimitFilter, configure_logging


# Fixtures
@pytest.fixture
def formatter():
    """Fixture providing a ColoredFormatter instance."""
    return ColoredFormatter("%(levelname)s: %(message)s")


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


def test_info_formatting(formatter, test_message):
    """Test INFO level message formatting."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg=test_message,
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)

    # INFO should use reset color (no color)
    assert "\033[0m" in formatted
    assert test_message in formatted
    assert "INFO" in formatted


def test_debug_formatting(formatter, test_message):
    """Test DEBUG level message formatting."""
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg=test_message,
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)

    # DEBUG should use reset color (no color)
    assert "\033[0m" in formatted
    assert test_message in formatted
    assert "DEBUG" in formatted


def test_warning_formatting(formatter, test_message):
    """Test WARNING level message formatting."""
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg=test_message,
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)

    # WARNING should use yellow/orange color
    assert "\033[33m" in formatted
    assert test_message in formatted
    assert "WARNING" in formatted
    # Should end with reset
    assert formatted.endswith("\033[0m")


def test_error_formatting(formatter, test_message):
    """Test ERROR level message formatting."""
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg=test_message,
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)

    # ERROR should use red color
    assert "\033[31m" in formatted
    assert test_message in formatted
    assert "ERROR" in formatted
    # Should end with reset
    assert formatted.endswith("\033[0m")


def test_critical_formatting(formatter, test_message):
    """Test CRITICAL level message formatting."""
    record = logging.LogRecord(
        name="test",
        level=logging.CRITICAL,
        pathname="test.py",
        lineno=1,
        msg=test_message,
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)

    # CRITICAL should use bold red color
    assert "\033[1;31m" in formatted
    assert test_message in formatted
    assert "CRITICAL" in formatted
    # Should end with reset
    assert formatted.endswith("\033[0m")


def test_color_codes_are_ansi():
    """Test that color codes are valid ANSI escape sequences."""
    # Test all defined colors
    for level_name, color_code in ColoredFormatter.COLORS.items():
        # ANSI color codes should match pattern \033[<number>m or \033[<number>;<number>m (for bold, etc.)
        assert re.match(r"\033\[[\d;]+m", color_code), f"Invalid ANSI color code for {level_name}"

    # Test reset code
    assert re.match(r"\033\[[\d;]+m", ColoredFormatter.RESET), "Invalid ANSI reset code"


def test_custom_format_string(test_message):
    """Test that custom format strings work correctly."""
    custom_formatter = ColoredFormatter("%(name)s - %(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="custom.logger",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg=test_message,
        args=(),
        exc_info=None,
    )
    formatted = custom_formatter.format(record)

    assert "custom.logger" in formatted
    assert "WARNING" in formatted
    assert test_message in formatted
    assert "\033[33m" in formatted  # Warning color


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


def test_first_warning_passes(rate_limit_filter):
    """Test that the first WARNING message passes through."""
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="First warning",
        args=(),
        exc_info=None,
    )
    assert rate_limit_filter.filter(record) is True


def test_duplicate_warning_within_interval_blocked(rate_limit_filter):
    """Test that duplicate WARNING messages within interval are blocked."""
    message = "Duplicate warning"

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
    assert rate_limit_filter.filter(record1) is True

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
    assert rate_limit_filter.filter(record2) is False


def test_warning_after_interval_passes():
    """Test that WARNING messages pass after the rate limit interval."""
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

    # Wait for interval to pass
    time.sleep(1.1)

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


def test_different_warnings_not_rate_limited(rate_limit_filter):
    """Test that different WARNING messages are not rate limited together."""
    # First warning
    record1 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="Warning A",
        args=(),
        exc_info=None,
    )
    assert rate_limit_filter.filter(record1) is True

    # Different warning should also pass
    record2 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=2,
        msg="Warning B",
        args=(),
        exc_info=None,
    )
    assert rate_limit_filter.filter(record2) is True


def test_custom_interval():
    """Test that custom interval seconds work correctly."""
    custom_filter = RateLimitFilter(interval_seconds=1)
    assert custom_filter.interval == 1

    long_filter = RateLimitFilter(interval_seconds=10)
    assert long_filter.interval == 10


def test_last_emitted_tracking(rate_limit_filter):
    """Test that the filter correctly tracks last emission times."""
    message1 = "Message 1"
    message2 = "Message 2"

    # Emit first message
    record1 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg=message1,
        args=(),
        exc_info=None,
    )
    rate_limit_filter.filter(record1)

    # Check that message1 is tracked
    assert message1 in rate_limit_filter.last_emitted

    # Emit second message
    record2 = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=2,
        msg=message2,
        args=(),
        exc_info=None,
    )
    rate_limit_filter.filter(record2)

    # Check that both messages are tracked
    assert message1 in rate_limit_filter.last_emitted
    assert message2 in rate_limit_filter.last_emitted

    # Timestamps should be different (though very close)
    assert rate_limit_filter.last_emitted[message1] <= rate_limit_filter.last_emitted[message2]


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
Integration Tests.

Tests that the filter and formatter work together in a logger.
"""


def test_filter_and_formatter_together():
    """Test that filter and formatter work together in a logger."""
    # Create a logger with both filter and formatter
    test_logger = logging.getLogger("test_integration")
    test_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    test_logger.handlers.clear()

    # Create handler with colored formatter
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))

    # Add rate limit filter
    rate_filter = RateLimitFilter(interval_seconds=1)
    handler.addFilter(rate_filter)

    test_logger.addHandler(handler)

    # Test that logger is set up correctly
    assert len(test_logger.handlers) == 1
    assert isinstance(test_logger.handlers[0].formatter, ColoredFormatter)

    # Clean up
    test_logger.handlers.clear()


def test_default_initialization():
    """Test that classes can be initialized with default parameters."""
    # ColoredFormatter with default format
    formatter = ColoredFormatter()
    assert formatter is not None

    # RateLimitFilter with default interval
    filter_obj = RateLimitFilter()
    assert filter_obj.interval == 5  # default is 5 seconds


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
    # Root logger is always set to DEBUG to ensure all messages are logged
    assert logger.level == logging.DEBUG

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
    """Test configure_logging with file logging enabled."""
    # Setup logger with file logging
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = configure_logging(logging_level="DEBUG", save_logs_to_file=True, log_dir=temp_dir)

        # Should return root logger
        assert logger is not None
        # Root logger is always set to DEBUG
        assert logger.level == logging.DEBUG

        # Should have two handlers (stream + file)
        assert len(logger.handlers) == 2

        # Check stream handler
        stream_handler = logger.handlers[0]
        assert isinstance(stream_handler, logging.StreamHandler)
        assert isinstance(stream_handler.formatter, ColoredFormatter)
        assert stream_handler.level == logging.DEBUG

        # Check file handler
        file_handler = logger.handlers[1]
        assert isinstance(file_handler, logging.FileHandler)
        assert file_handler.level == logging.DEBUG

        # Verify log file was created
        log_files = [f for f in os.listdir(temp_dir) if f.startswith("isaaclab_")]
        assert len(log_files) == 1


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
        # Root logger is always set to DEBUG to ensure all messages are logged
        assert logger.level == logging.DEBUG
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

    # Root logger is always set to DEBUG
    assert logger.level == logging.DEBUG

    # Should have file handler
    assert len(logger.handlers) == 2
    file_handler = logger.handlers[1]
    assert isinstance(file_handler, logging.FileHandler)

    # File should be in temp directory
    log_file_path = file_handler.baseFilename
    assert os.path.dirname(log_file_path) == os.path.join(tempfile.gettempdir(), "isaaclab", "logs")
    assert os.path.basename(log_file_path).startswith("isaaclab_")

    # Cleanup
    if os.path.exists(log_file_path):
        os.remove(log_file_path)


def test_configure_logging_custom_log_dir():
    """Test configure_logging with custom log directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_log_dir = os.path.join(temp_dir, "custom_logs")

        logger = configure_logging(logging_level="INFO", save_logs_to_file=True, log_dir=custom_log_dir)

        # Custom directory should be created
        assert os.path.exists(custom_log_dir)
        assert os.path.isdir(custom_log_dir)

        # Root logger is always set to DEBUG
        assert logger.level == logging.DEBUG

        # Log file should be in custom directory
        file_handler = logger.handlers[1]
        assert isinstance(file_handler, logging.FileHandler)
        log_file_path = file_handler.baseFilename
        assert os.path.dirname(log_file_path) == custom_log_dir


def test_configure_logging_log_file_format():
    """Test that log file has correct timestamp format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = configure_logging(logging_level="INFO", save_logs_to_file=True, log_dir=temp_dir)

        # Root logger is always set to DEBUG
        assert logger.level == logging.DEBUG

        # Get log file name
        file_handler = logger.handlers[1]
        assert isinstance(file_handler, logging.FileHandler)
        log_file_path = file_handler.baseFilename
        log_filename = os.path.basename(log_file_path)

        # Check filename format: isaaclab_YYYY-MM-DD_HH-MM-SS.log
        pattern = r"isaaclab_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log"
        assert re.match(pattern, log_filename), f"Log filename {log_filename} doesn't match expected pattern"


def test_configure_logging_file_formatter():
    """Test that file handler has more detailed formatter than stream handler."""
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = configure_logging(logging_level="INFO", save_logs_to_file=True, log_dir=temp_dir)

        # Root logger is always set to DEBUG
        assert logger.level == logging.DEBUG

        stream_handler = logger.handlers[0]
        file_handler = logger.handlers[1]

        # Stream formatter should exist and be ColoredFormatter
        assert stream_handler.formatter is not None
        assert isinstance(stream_handler.formatter, ColoredFormatter)
        stream_format = stream_handler.formatter._fmt  # type: ignore
        assert stream_format is not None
        assert "%(asctime)s" in stream_format
        assert "%(filename)s" in stream_format

        # File formatter should exist and include line numbers
        assert file_handler.formatter is not None
        assert isinstance(file_handler.formatter, logging.Formatter)
        file_format = file_handler.formatter._fmt  # type: ignore
        assert file_format is not None
        assert "%(asctime)s" in file_format
        assert "%(lineno)d" in file_format

        # File handler should always use DEBUG level
        assert file_handler.level == logging.DEBUG


def test_configure_logging_multiple_calls():
    """Test that multiple configure_logging calls properly cleanup."""
    # First setup
    logger1 = configure_logging(logging_level="INFO", save_logs_to_file=False)
    handler_count_1 = len(logger1.handlers)

    # Second setup should remove previous handlers
    logger2 = configure_logging(logging_level="DEBUG", save_logs_to_file=False)
    handler_count_2 = len(logger2.handlers)

    # Should be same logger (root logger)
    assert logger1 is logger2

    # Should have same number of handlers (old ones removed)
    assert handler_count_1 == handler_count_2 == 1


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
