# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Description of a single replication layout, consumed by :func:`~isaaclab.cloner.replicate`."""

    sources: tuple[str, ...]
    """Source prim paths, one per replication row."""

    destinations: tuple[str, ...]
    """Destination path templates with ``"{}"`` for the env id, one per row."""

    clone_mask: torch.Tensor
    """Bool tensor ``[len(sources), num_clones]``; ``True`` if env ``j`` comes from row ``i``."""

    env_ids: torch.Tensor | None = None
    """Long tensor ``[num_clones]`` of target env ids.

    Optional for plans used only with :func:`~isaaclab.cloner.iter_clone_plan_matches` or
    :func:`~isaaclab.cloner.resolve_clone_plan_source`; required by :func:`~isaaclab.cloner.replicate`.
    """

    positions: torch.Tensor | None = None
    """Per-env world positions [m], shape ``[num_clones, 3]``, or ``None``."""

    cfg_rows: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """``id(cfg)`` to the row indices the cfg owns."""

    @classmethod
    def from_env_0(
        cls,
        source: str,
        destination: str,
        num_clones: int,
        device: str,
        positions: torch.Tensor | None = None,
    ) -> ClonePlan:
        """Build a single-source clone plan that targets every env from one source row.

        Auto-populates :attr:`cfg_rows` from
        :data:`~isaaclab.cloner.REPLICATION_QUEUE`, including only cfgs whose
        ``prim_path`` falls under the env-root prefix of ``destination``. Must be
        called *after* all asset constructors have run, so their replication entries
        are already in the queue; otherwise those assets will be skipped by the
        subsequent :func:`~isaaclab.cloner.replicate` call.

        Args:
            source: Source prim path (typically ``/World/envs/env_0``).
            destination: Destination template with ``"{}"`` for the env id.
            num_clones: Number of target envs.
            device: Torch device for the mask and env id buffers.
            positions: Optional per-env world positions [m], shape ``[num_clones, 3]``.
        """
        from .replicate_session import REPLICATION_QUEUE  # noqa: PLC0415

        prefix, _, _ = destination.partition("{}")
        cfg_rows: dict[int, tuple[int, ...]] = {
            id(cfg): (0,) for cfg, _ in REPLICATION_QUEUE if cfg.prim_path.startswith(prefix)
        }
        return cls(
            sources=(source,),
            destinations=(destination,),
            clone_mask=torch.ones((1, num_clones), dtype=torch.bool, device=device),
            env_ids=torch.arange(num_clones, dtype=torch.long, device=device),
            positions=positions,
            cfg_rows=cfg_rows,
        )
