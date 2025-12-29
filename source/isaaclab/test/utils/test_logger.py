# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
import re
import time

import pytest

from isaaclab.utils.logger import ColoredFormatter, RateLimitFilter


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

    # CRITICAL should use red color
    assert "\033[31m" in formatted
    assert test_message in formatted
    assert "CRITICAL" in formatted
    # Should end with reset
    assert formatted.endswith("\033[0m")


def test_color_codes_are_ansi():
    """Test that color codes are valid ANSI escape sequences."""
    # Test all defined colors
    for level_name, color_code in ColoredFormatter.COLORS.items():
        # ANSI color codes should match pattern \033[<number>m
        assert re.match(r"\033\[\d+m", color_code), f"Invalid ANSI color code for {level_name}"

    # Test reset code
    assert re.match(r"\033\[\d+m", ColoredFormatter.RESET), "Invalid ANSI reset code"


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
