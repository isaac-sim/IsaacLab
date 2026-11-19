# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Application-owned image generation contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class ImageTransferStream(Protocol):
    """One batched generation stream with temporal state for each camera view.

    Controls are uint8 THWC three-channel image sequences, one per view, valid until :meth:`step`
    returns: the processor reuses their storage for later captures, so a stream that keeps a chunk
    copies it. Returned images must be uint8 THWC sRGB with the same sequence lengths and image
    size. Calls run on the current Torch stream; returned tensors must be ready on that stream.
    Reset rows discard all episode state and start again from the given seeds.
    """

    def step(
        self, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]
    ) -> list[torch.Tensor] | None:
        """Consume one chunk per view and return the corresponding images.

        A stream may instead return None and deliver the images later through a callback it was given.
        """
        ...

    def close(self) -> None:
        """Release the stream's state. Repeated calls must be safe."""
        ...


class ImageTransferModel(Protocol):
    """Shared, stateless model resource constructed from a :class:`~isaaclab.sim.BackendCfg`.

    :meth:`~isaaclab.sim.SimulationContext.get_or_create_backend` returns one model for equal
    configurations, so the model must keep per-camera state in the streams it opens.
    """

    def open_stream(self, num_views: int, seeds: tuple[int, ...]) -> ImageTransferStream:
        """Open a stream for ``num_views`` camera views, seeded with one initial seed per view."""
        ...

    def close(self) -> None:
        """Release the model. Repeated calls must be safe."""
        ...
