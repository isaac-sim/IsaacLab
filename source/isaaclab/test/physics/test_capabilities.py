# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the capability registry and wire-up validator.

These tests exercise the framework in isolation — no Isaac Sim, no real
provider, no real renderer. A test-only ``_FakeProvider`` and a couple of
test-only consumer classes are sufficient because the framework's contract
is purely structural.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

import pytest

from isaaclab.physics import (
    BaseSceneDataProvider,
    CapabilityRequirementError,
    GpuTransformBuffer,
    UsdFabric,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@runtime_checkable
class _CustomCap(Protocol):
    """A customer-defined capability for cross-package extensibility tests."""

    def acme_method(self) -> int: ...


class _FakeGpuHandle:
    def get_body_transforms(self, target_format, **_kwargs):
        return None

    def get_source_format(self):
        return None


class _FakeFabricHandle:
    def ensure_current(self, stream=None):
        return None


class _FakeCustomHandle:
    def acme_method(self) -> int:
        return 42


class _FakeProvider(BaseSceneDataProvider):
    """Minimal concrete provider for framework tests.

    The base class declares many abstract methods inherited from earlier
    designs; we stub them out with no-ops because none are exercised here.
    """

    def __init__(self):
        super().__init__()

    def update(self) -> None: ...
    def get_newton_model(self): return None
    def get_newton_state(self): return None
    def get_usd_stage(self): return None
    def get_metadata(self): return {}
    def get_transforms(self): return None
    def get_velocities(self): return None
    def get_contacts(self): return None
    def get_camera_transforms(self): return None


class _ConsumerWantingGpu:
    required_capabilities: ClassVar[tuple[type, ...]] = (GpuTransformBuffer,)


class _ConsumerWantingUsd:
    required_capabilities: ClassVar[tuple[type, ...]] = (UsdFabric,)


class _ConsumerWantingEither:
    required_one_of: ClassVar[tuple[tuple[type, ...], ...]] = (
        (UsdFabric, GpuTransformBuffer),
    )


class _ConsumerWantingBoth:
    required_capabilities: ClassVar[tuple[type, ...]] = (
        GpuTransformBuffer,
        UsdFabric,
    )


class _ConsumerWantingCustom:
    required_capabilities: ClassVar[tuple[type, ...]] = (_CustomCap,)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_empty_provider_lists_no_capabilities():
    p = _FakeProvider()
    assert p.list_capabilities() == frozenset()
    assert p.get_capability(GpuTransformBuffer) is None


def test_registered_capability_is_returned():
    p = _FakeProvider()
    handle = _FakeGpuHandle()
    p._register_capability(GpuTransformBuffer, handle)

    assert p.get_capability(GpuTransformBuffer) is handle
    assert p.list_capabilities() == frozenset({GpuTransformBuffer})


def test_re_registration_replaces_handle():
    p = _FakeProvider()
    first = _FakeGpuHandle()
    second = _FakeGpuHandle()
    p._register_capability(GpuTransformBuffer, first)
    p._register_capability(GpuTransformBuffer, second)
    assert p.get_capability(GpuTransformBuffer) is second


def test_get_first_capability_walks_in_order():
    p = _FakeProvider()
    p._register_capability(GpuTransformBuffer, _FakeGpuHandle())

    # Preferred missing → falls back to second.
    result = p.get_first_capability(UsdFabric, GpuTransformBuffer)
    assert result is not None
    cap_type, handle = result
    assert cap_type is GpuTransformBuffer
    assert isinstance(handle, _FakeGpuHandle)


def test_get_first_capability_returns_none_when_all_missing():
    p = _FakeProvider()
    assert p.get_first_capability(UsdFabric, GpuTransformBuffer) is None


def test_runtime_checkable_protocol_isinstance():
    """The protocols are runtime_checkable; isinstance works on duck types."""
    handle = _FakeGpuHandle()
    assert isinstance(handle, GpuTransformBuffer)


def test_custom_capability_routes_through_registry():
    """Customer-defined caps work without framework changes."""
    p = _FakeProvider()
    handle = _FakeCustomHandle()
    p._register_capability(_CustomCap, handle)
    assert p.get_capability(_CustomCap) is handle
    assert _CustomCap in p.list_capabilities()


# ---------------------------------------------------------------------------
# Consumer registration and wire-up validation
# ---------------------------------------------------------------------------


def test_validate_passes_with_no_consumers():
    p = _FakeProvider()
    p.validate_consumer_capabilities()  # no consumers → no error


def test_validate_passes_when_required_satisfied():
    p = _FakeProvider()
    p._register_capability(GpuTransformBuffer, _FakeGpuHandle())
    p.register_consumer(_ConsumerWantingGpu())
    p.validate_consumer_capabilities()


def test_validate_fails_with_missing_required():
    p = _FakeProvider()
    p.register_consumer(_ConsumerWantingGpu())
    with pytest.raises(CapabilityRequirementError) as excinfo:
        p.validate_consumer_capabilities()
    assert "GpuTransformBuffer" in str(excinfo.value)
    assert "ConsumerWantingGpu" in str(excinfo.value)


def test_validate_message_lists_offered_capabilities():
    p = _FakeProvider()
    p._register_capability(UsdFabric, _FakeFabricHandle())
    p.register_consumer(_ConsumerWantingGpu())
    with pytest.raises(CapabilityRequirementError) as excinfo:
        p.validate_consumer_capabilities()
    assert "UsdFabric" in str(excinfo.value)


def test_validate_consolidates_multiple_failures():
    """A single error reports failures for every offending consumer."""
    p = _FakeProvider()
    p.register_consumer(_ConsumerWantingGpu())
    p.register_consumer(_ConsumerWantingUsd())
    with pytest.raises(CapabilityRequirementError) as excinfo:
        p.validate_consumer_capabilities()
    msg = str(excinfo.value)
    assert "ConsumerWantingGpu" in msg
    assert "ConsumerWantingUsd" in msg


def test_required_one_of_passes_when_any_present():
    p = _FakeProvider()
    p._register_capability(GpuTransformBuffer, _FakeGpuHandle())
    p.register_consumer(_ConsumerWantingEither())
    p.validate_consumer_capabilities()


def test_required_one_of_fails_when_all_missing():
    p = _FakeProvider()
    p.register_consumer(_ConsumerWantingEither())
    with pytest.raises(CapabilityRequirementError) as excinfo:
        p.validate_consumer_capabilities()
    assert "one-of" in str(excinfo.value)


def test_required_capabilities_are_all_required():
    p = _FakeProvider()
    p._register_capability(GpuTransformBuffer, _FakeGpuHandle())
    # UsdFabric missing
    p.register_consumer(_ConsumerWantingBoth())
    with pytest.raises(CapabilityRequirementError) as excinfo:
        p.validate_consumer_capabilities()
    assert "UsdFabric" in str(excinfo.value)
    assert "GpuTransformBuffer" not in str(excinfo.value).split("missing required:")[1].split("]")[0]


def test_register_consumer_invalidates_prior_validation():
    """A new registration after validate() must force re-validation."""
    p = _FakeProvider()
    p._register_capability(GpuTransformBuffer, _FakeGpuHandle())
    p.register_consumer(_ConsumerWantingGpu())
    p.validate_consumer_capabilities()
    assert p._capabilities_validated is True

    p.register_consumer(_ConsumerWantingUsd())
    assert p._capabilities_validated is False


def test_validate_if_needed_is_idempotent():
    p = _FakeProvider()
    p._register_capability(GpuTransformBuffer, _FakeGpuHandle())
    p.register_consumer(_ConsumerWantingGpu())
    p._validate_consumers_if_needed()
    # Second call is a no-op even if we mutate the registry behind its back.
    p._capabilities.pop(GpuTransformBuffer)
    p._validate_consumers_if_needed()  # no error — already validated


def test_consumer_registry_uses_weak_references():
    """Dropped consumers do not leak into validation."""
    import gc

    p = _FakeProvider()
    p._register_capability(GpuTransformBuffer, _FakeGpuHandle())

    consumer = _ConsumerWantingUsd()
    p.register_consumer(consumer)
    del consumer
    gc.collect()

    p.validate_consumer_capabilities()  # no error: consumer was collected


def test_custom_capability_validates_like_built_ins():
    p = _FakeProvider()
    p._register_capability(_CustomCap, _FakeCustomHandle())
    p.register_consumer(_ConsumerWantingCustom())
    p.validate_consumer_capabilities()
