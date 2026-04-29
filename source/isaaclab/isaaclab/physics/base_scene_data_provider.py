# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider interface for visualizers and renderers."""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from .capabilities._validator import validate_consumer_capabilities

T = TypeVar("T")


class BaseSceneDataProvider(ABC):
    """Backend-agnostic scene data provider interface.

    The SDP acts as a central hub that bridges simulator data to renderers
    and visualizers. It does not own simulation data — it provides
    format-negotiated access to it.

    Provider services are exposed as **capabilities**: typed protocol
    handles registered by the provider and queried by consumers. See
    :doc:`/source/refs/adr/0002-capability-based-provider-extensibility`.
    """

    def __init__(self) -> None:
        self._capabilities: dict[type, Any] = {}
        self._consumers: weakref.WeakSet = weakref.WeakSet()
        self._capabilities_validated = False

    # ------------------------------------------------------------------
    # Capability registry
    # ------------------------------------------------------------------

    def get_capability(self, cap_type: type[T]) -> T | None:
        """Return the registered handle for *cap_type*, or ``None``.

        Args:
            cap_type: Capability protocol class to look up.

        Returns:
            The handle registered for *cap_type*, or ``None`` when this
            provider does not offer that capability.
        """
        return self._capabilities.get(cap_type)

    def list_capabilities(self) -> frozenset[type]:
        """Return the set of capability types this provider offers."""
        return frozenset(self._capabilities)

    def get_first_capability(self, *cap_types: type) -> tuple[type, Any] | None:
        """Return the first registered capability among *cap_types*.

        Walks the arguments in order and returns the first match.

        Args:
            *cap_types: Capability protocol classes in preference order.

        Returns:
            A ``(cap_type, handle)`` tuple for the first match, or ``None``
            when none of the requested capabilities are registered.
        """
        for cap_type in cap_types:
            handle = self._capabilities.get(cap_type)
            if handle is not None:
                return cap_type, handle
        return None

    def _register_capability(self, cap_type: type[T], handle: T) -> None:
        """Register *handle* as the implementation of *cap_type*.

        Subclasses call this from ``__init__`` for every capability they
        offer. Re-registering a capability replaces the previous handle.
        """
        self._capabilities[cap_type] = handle

    # ------------------------------------------------------------------
    # Consumer registry and wire-up validation
    # ------------------------------------------------------------------

    def register_consumer(self, consumer: Any) -> None:
        """Register a consumer (renderer or visualizer) with this provider.

        Consumers self-register so the provider can validate their
        capability requirements at wire-up. The consumer is held by weak
        reference; the provider does not extend its lifetime.
        """
        self._consumers.add(consumer)
        # Re-validate next update; new consumer may have unmet requirements.
        self._capabilities_validated = False

    def validate_consumer_capabilities(self) -> None:
        """Check that every registered consumer's requirements are met.

        Raises:
            CapabilityRequirementError: One or more consumers have unmet
                ``required_capabilities`` or ``required_one_of`` declarations.
        """
        validate_consumer_capabilities(self, list(self._consumers))
        self._capabilities_validated = True

    def _validate_consumers_if_needed(self) -> None:
        """Idempotent validator hook for use from per-frame ``update()``."""
        if not self._capabilities_validated:
            self.validate_consumer_capabilities()

    # ------------------------------------------------------------------
    # Provider lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def update(self) -> None:
        """Refresh any cached scene data (full model/state)."""
        raise NotImplementedError

    @abstractmethod
    def get_usd_stage(self) -> Any | None:
        """Return USD stage handle when available.

        Kept on the base class because USD stage access is orthogonal to
        capability registration — consumers that need only the stage (not
        per-frame USD freshness) use this directly. For per-frame freshness
        guarantees, request the :class:`UsdFabric` capability instead.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return backend metadata (num_envs, gravity, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Return per-camera, per-env transforms, if supported."""
        raise NotImplementedError
