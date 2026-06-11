# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ServiceLocator."""

import pytest

from isaaclab.sim.service_locator import ServiceLocator

# -- Dummy service helpers --


class _DummyServiceWithClose:
    """Service with a callable close() method."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _DummyServiceWithoutClose:
    """Service without a close() method."""

    pass


class _DummyServiceWithCloseProperty:
    """Service where 'close' exists but is not callable."""

    close = 42  # attribute, not a method


class _DummyServiceThatThrows:
    """Service whose close() raises an exception."""

    def close(self):
        raise RuntimeError("close failed")


# -- Fixtures --


@pytest.fixture
def locator():
    """Provide a fresh ServiceLocator for each test."""
    return ServiceLocator()


# -- Tests --


def test_get_returns_none_when_unregistered(locator):
    assert locator[_DummyServiceWithClose] is None


def test_set_and_get(locator):
    svc = _DummyServiceWithClose()
    locator[_DummyServiceWithClose] = svc
    assert locator[_DummyServiceWithClose] is svc


def test_contains(locator):
    assert _DummyServiceWithClose not in locator
    locator[_DummyServiceWithClose] = _DummyServiceWithClose()
    assert _DummyServiceWithClose in locator


def test_del_closes_service(locator):
    svc = _DummyServiceWithClose()
    locator[_DummyServiceWithClose] = svc
    del locator[_DummyServiceWithClose]
    assert svc.closed
    assert locator[_DummyServiceWithClose] is None


def test_del_without_close_method(locator):
    """del on a service without close() should not raise."""
    locator[_DummyServiceWithoutClose] = _DummyServiceWithoutClose()
    del locator[_DummyServiceWithoutClose]
    assert locator[_DummyServiceWithoutClose] is None


def test_del_with_non_callable_close_property(locator):
    """del on a service where close is a property (not callable) should not raise."""
    svc = _DummyServiceWithCloseProperty()
    locator[_DummyServiceWithCloseProperty] = svc
    del locator[_DummyServiceWithCloseProperty]
    assert locator[_DummyServiceWithCloseProperty] is None


def test_del_missing_raises_key_error(locator):
    with pytest.raises(KeyError):
        del locator[_DummyServiceWithClose]


def test_pop_returns_without_closing(locator):
    svc = _DummyServiceWithClose()
    locator[_DummyServiceWithClose] = svc
    popped = locator.pop(_DummyServiceWithClose)
    assert popped is svc
    assert not svc.closed
    assert locator[_DummyServiceWithClose] is None


def test_pop_missing_returns_none(locator):
    assert locator.pop(_DummyServiceWithClose) is None


def test_close_all(locator):
    svc1 = _DummyServiceWithClose()
    svc2 = _DummyServiceWithoutClose()
    locator[_DummyServiceWithClose] = svc1
    locator[_DummyServiceWithoutClose] = svc2
    errors: list[Exception] = []
    locator.close_all(caught_exceptions=errors)
    assert svc1.closed
    assert not errors
    assert locator[_DummyServiceWithClose] is None
    assert locator[_DummyServiceWithoutClose] is None


def test_close_all_skips_non_callable_close(locator):
    """close_all does not crash on services with non-callable close attribute."""
    locator[_DummyServiceWithCloseProperty] = _DummyServiceWithCloseProperty()
    errors: list[Exception] = []
    locator.close_all(caught_exceptions=errors)
    assert not errors
    assert locator[_DummyServiceWithCloseProperty] is None


def test_close_all_collects_exceptions(locator):
    """Exceptions are collected and all services still get closed."""
    svc_ok = _DummyServiceWithClose()
    locator[_DummyServiceWithClose] = svc_ok
    locator[_DummyServiceThatThrows] = _DummyServiceThatThrows()
    errors: list[Exception] = []
    locator.close_all(caught_exceptions=errors)
    assert svc_ok.closed
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert locator[_DummyServiceWithClose] is None
    assert locator[_DummyServiceThatThrows] is None


def test_multiple_service_types(locator):
    svc1 = _DummyServiceWithClose()
    svc2 = _DummyServiceWithoutClose()
    locator[_DummyServiceWithClose] = svc1
    locator[_DummyServiceWithoutClose] = svc2
    assert locator[_DummyServiceWithClose] is svc1
    assert locator[_DummyServiceWithoutClose] is svc2


def test_base_class_key(locator):
    """Can register under a base class and retrieve by it."""

    class Base:
        pass

    class Impl(Base):
        pass

    impl = Impl()
    locator[Base] = impl
    assert locator[Base] is impl
