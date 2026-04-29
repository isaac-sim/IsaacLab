# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Built-in capability protocols.

Each capability is a :func:`typing.runtime_checkable` :class:`typing.Protocol`
identifying a specific provider service. Customers may define their own
protocols in their own packages following the same pattern; identity is
by Python ``type``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import warp as wp

from ..scene_data_types import (
    MatrixLayout,
    QuaternionConvention,
    TransformData,
    TransformFormat,
)


@runtime_checkable
class GpuTransformBuffer(Protocol):
    """Typed GPU-buffer access to body transforms.

    Consumers declare a target :class:`TransformFormat` and the provider
    returns a typed :class:`TransformData` with converted data, or a
    zero-copy passthrough when the requested format matches the
    simulator's native format.

    This is the mandatory baseline capability — every Scene Data Provider
    must register an implementation, so every consumer can fall back to
    typed buffer pull when its preferred capability is unavailable.
    """

    def get_body_transforms(
        self,
        target_format: TransformFormat,
        *,
        env_ids: list[int] | None = None,
        quat_convention: QuaternionConvention = QuaternionConvention.XYZW,
        matrix_layout: MatrixLayout = MatrixLayout.ROW_MAJOR,
        double_precision: bool = False,
        stream: wp.Stream | None = None,
        allow_passthrough: bool = True,
        index_map: wp.array | None = None,
    ) -> TransformData | None:
        """Return body transforms in the requested format.

        Args:
            target_format: Desired output transform representation.
            env_ids: Optional environment subset to include.
            quat_convention: Quaternion component ordering for
                :attr:`TransformFormat.VEC3_QUAT` output.
            matrix_layout: Matrix memory layout for
                :attr:`TransformFormat.VEC3_MAT33` and
                :attr:`TransformFormat.MAT44` output.
            double_precision: Use 64-bit floats for
                :attr:`TransformFormat.MAT44` output.
            stream: CUDA stream for deferred kernel execution.
            allow_passthrough: If ``True`` and formats match, return the
                source data directly without copying.
            index_map: Optional index remapping array for subset scatter
                writes.

        Returns:
            Typed transform data, or ``None`` if not supported by this
            provider.
        """
        ...

    def get_source_format(self) -> TransformFormat | None:
        """Return the simulator's native transform format.

        Consumers can use this to request a passthrough-compatible format
        and avoid conversions entirely.

        Returns:
            The native transform format, or ``None`` if unknown.
        """
        ...


@runtime_checkable
class UsdFabric(Protocol):
    """USD Fabric channel — guarantees prim attributes reflect the current
    physics generation.

    Providers that natively write Fabric (e.g. PhysX via the Tensor API)
    implement :meth:`ensure_current` as a fast no-op. Providers that do
    not write Fabric natively (e.g. Newton) implement it by running their
    sync-to-USD bridge.
    """

    def ensure_current(self, stream: wp.Stream | None = None) -> None:
        """Ensure USD Fabric attributes reflect the current physics state.

        Args:
            stream: CUDA stream for deferred work, when applicable.
        """
        ...
