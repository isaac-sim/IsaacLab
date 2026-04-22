# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider interface for visualizers and renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import warp as wp

from .scene_data_types import (
    MatrixLayout,
    QuaternionConvention,
    TransformData,
    TransformFormat,
)


class BaseSceneDataProvider(ABC):
    """Backend-agnostic scene data provider interface.

    The SDP acts as a central hub that bridges simulator data to renderers
    and visualizers. It does not own simulation data—it provides
    format-negotiated access to it.

    Consumers should prefer the typed :meth:`get_body_transforms` API over
    the legacy :meth:`get_transforms` method.
    """

    # ------------------------------------------------------------------
    # Existing abstract methods (backward compatible)
    # ------------------------------------------------------------------

    @abstractmethod
    def update(self) -> None:
        """Refresh any cached scene data (full model/state)."""
        raise NotImplementedError

    @abstractmethod
    def get_newton_model(self) -> Any | None:
        """Return Newton model handle when available."""
        raise NotImplementedError

    @abstractmethod
    def get_newton_state(self) -> Any | None:
        """Return Newton state handle when available (full state)."""
        raise NotImplementedError

    @abstractmethod
    def get_usd_stage(self) -> Any | None:
        """Return USD stage handle when available."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return backend metadata (num_envs, gravity, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def get_transforms(self) -> dict[str, Any] | None:
        """Return body transforms, if supported."""
        raise NotImplementedError

    @abstractmethod
    def get_velocities(self) -> dict[str, Any] | None:
        """Return body velocities, if supported."""
        raise NotImplementedError

    @abstractmethod
    def get_contacts(self) -> dict[str, Any] | None:
        """Return contacts, if supported."""
        raise NotImplementedError

    @abstractmethod
    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Return per-camera, per-env transforms, if supported."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Typed transform API
    # ------------------------------------------------------------------

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

        Consumers declare what format they need; the provider converts from
        the simulator's native format using GPU-only Warp kernels. Conversion
        results are cached per frame and reused if multiple consumers request
        the same format.

        When the simulator's native format matches the requested format and
        *allow_passthrough* is ``True``, the returned :class:`TransformData`
        may reference the simulator's own GPU buffers (zero-copy passthrough).
        The consumer must not mutate the returned data in this case.

        Args:
            target_format: Desired output transform representation.
            env_ids: Optional environment subset. When provided, only
                transforms for the specified environments are included.
            quat_convention: Quaternion component ordering for
                :attr:`TransformFormat.VEC3_QUAT` output.
            matrix_layout: Matrix memory layout for
                :attr:`TransformFormat.VEC3_MAT33` and
                :attr:`TransformFormat.MAT44` output.
            double_precision: Use 64-bit floats for
                :attr:`TransformFormat.MAT44` output.
            stream: CUDA stream for deferred kernel execution. When provided,
                conversion kernels are enqueued on this stream instead of
                the default stream.
            allow_passthrough: If ``True`` and formats match, return the
                source data directly without copying. The consumer must not
                mutate the result.
            index_map: Optional index remapping array for subset scatter
                writes. When provided, ``output[i]`` is written from
                ``source[index_map[i]]``.

        Returns:
            Typed transform data, or ``None`` if not supported by this
            provider.
        """
        return None

    def get_source_format(self) -> TransformFormat | None:
        """Return the simulator's native transform format.

        Consumers can use this to request a passthrough-compatible format and
        avoid conversions entirely.

        Returns:
            The native transform format, or ``None`` if unknown.
        """
        return None
