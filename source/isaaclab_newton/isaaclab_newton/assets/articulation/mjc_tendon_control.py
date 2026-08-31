# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private MuJoCo tendon control adapter for the Newton backend.

Tendons are a MuJoCo concept, so a fixed tendon is driven by whichever MuJoCo actuator transmits
to it. :class:`~newton.selection.ArticulationView` exposes the ``mujoco:actuator`` frequency, so
this maps each fixed tendon to its actuator column and lets the view own the world and instance
layout; it owns the per-step write that carries buffered targets into ``mujoco.ctrl``.

:class:`MjcTendonControl` holds the articulation it drives, so it reuses that asset's index
resolution and shape checking rather than restating them -- the same arrangement
:class:`~isaaclab_newton.assets.articulation.actuator_control.NewtonActuatorControl` uses for
actuators. The command buffer itself stays on
:class:`~isaaclab_newton.assets.ArticulationData`, beside the other fixed-tendon buffers.

Nothing here is exported; :mod:`isaaclab_newton.assets.articulation.articulation` is the only
importer. That is what lets the whole module be deleted once Newton exposes the mapping itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton.solvers import SolverMuJoCo

from isaaclab_newton.assets import kernels as shared_kernels

from . import kernels as articulation_kernels
from .mjc_view_compat import ensure_custom_frequency_api

if TYPE_CHECKING:
    from newton import Control, Model
    from newton.selection import ArticulationView

    from .articulation import Articulation

logger = logging.getLogger(__name__)


class MjcTendonControl:
    """Drives an articulation's fixed tendons through MuJoCo's native tendon actuators.

    Created by :meth:`~isaaclab_newton.assets.Articulation._process_tendons` when the model carries
    at least one directly-actuated fixed tendon. Command tendons through the articulation's
    backend-neutral
    :meth:`~isaaclab.assets.articulation.BaseArticulation.set_fixed_tendon_position_target_index`
    rather than through this internal adapter.
    """

    @classmethod
    def create(cls, articulation: Articulation, model: Model) -> MjcTendonControl | None:
        """Build the adapter, or None when no actuator transmits to any of the tendons.

        Args:
            articulation: Newton articulation owning the fixed tendons.
            model: Newton model the articulation's view selects from.

        Returns:
            The adapter, or None when the articulation's tendons are all passive.
        """
        view = ensure_custom_frequency_api(articulation.root_view, model)
        columns = resolve_fixed_tendon_actuator_columns(view, model)
        if columns is None or not bool((columns >= 0).any()):
            return None
        return cls(articulation, columns, view)

    def __init__(self, articulation: Articulation, actuator_columns: np.ndarray, view: ArticulationView):
        """Bind a tendon-to-actuator mapping to the articulation it drives.

        Args:
            articulation: Newton articulation owning the fixed tendons.
            actuator_columns: ``mujoco:actuator`` column per fixed tendon, ``-1`` where no actuator
                transmits to that tendon. Shape is ``(num_fixed_tendons,)``.
            view: Articulation view carrying the custom-frequency API, from
                :func:`~isaaclab_newton.assets.articulation.mjc_view_compat.ensure_custom_frequency_api`.
        """
        self._articulation = articulation
        self._actuator_columns = wp.array(actuator_columns, dtype=wp.int32, device=articulation.device)
        self._view = view
        if view.count_per_world != 1:
            raise NotImplementedError(
                f"The articulation view reports {view.count_per_world} instances per world; the"
                " tendon command buffer is written one instance per environment. Scattering into"
                " the other instances needs the view's own layout, not arithmetic here."
            )
        # View-shaped scratch the per-step write scatters into, allocated once. Newton accepts
        # exactly ``(world_count, count_per_world, actuator_count)`` for ``mujoco.ctrl`` and owns
        # the mapping onto the flat control array.
        self._commands = wp.zeros(
            (view.world_count, view.count_per_world, view.custom_frequency_counts["mujoco:actuator"]),
            dtype=wp.float32,
            device=articulation.device,
        )
        # Name the passive tendons once here rather than per command: the IDs reaching
        # :meth:`set_position_target` are a device array, so checking them there would synchronize
        # the GPU on every step.
        passive = [
            name
            for name, commandable in zip(articulation.fixed_tendon_names, actuator_columns >= 0, strict=True)
            if not commandable
        ]
        if passive:
            logger.warning(
                "Fixed tendons %s have no direct MuJoCo position actuator, so commanding them has no effect."
                " Author an actuator whose transmission is each tendon to drive them.",
                passive,
            )

    def set_position_target(
        self,
        *,
        target: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Buffer fixed-tendon position targets in the articulation's fixed-tendon index space.

        This does not reach the simulation; :meth:`write_data_to_sim` carries the buffer into
        ``mujoco.ctrl`` on the articulation's usual per-step write.

        Args:
            target: Target tendon length [m or rad, depending on the spanned joints' type].
                Shape is ``(len(env_ids), len(fixed_tendon_ids))``.
            fixed_tendon_ids: The tendon indices to command. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        articulation = self._articulation
        env_ids = articulation._resolve_env_ids(env_ids)
        fixed_tendon_ids = articulation._resolve_fixed_tendon_ids(fixed_tendon_ids)
        articulation.assert_shape_and_dtype(target, (env_ids.shape[0], fixed_tendon_ids.shape[0]), wp.float32, "target")
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices_kernel(env_ids, fixed_tendon_ids),
            dim=(env_ids.shape[0], fixed_tendon_ids.shape[0]),
            inputs=[target, env_ids, fixed_tendon_ids],
            outputs=[articulation.data._fixed_tendon_position_target],
            device=articulation.device,
        )

    def set_position_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Buffer fixed-tendon position targets, selecting tendons and environments by mask.

        The mask form exists so the write can be captured in a CUDA graph: resolving a mask to
        indices would read the mask on the host and break capture, so the masks reach the kernel
        as they are.

        Args:
            target: Target tendon length [m or rad, depending on the spanned joints' type].
                Shape is ``(num_instances, num_fixed_tendons)``.
            fixed_tendon_mask: Fixed tendon mask. If None, every fixed tendon is commanded.
            env_mask: Environment mask. If None, every instance is commanded.
        """
        articulation = self._articulation
        buffer = articulation.data._fixed_tendon_position_target
        articulation.assert_shape_and_dtype(target, buffer.shape, wp.float32, "target")
        env_mask = articulation._resolve_mask(env_mask, articulation._ALL_ENV_MASK)
        fixed_tendon_mask = articulation._resolve_mask(fixed_tendon_mask, articulation._ALL_FIXED_TENDON_MASK)
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=buffer.shape,
            inputs=[target, env_mask, fixed_tendon_mask],
            outputs=[buffer],
            device=articulation.device,
        )

    def write_data_to_sim(self, control: Control) -> None:
        """Scatter the buffered targets into the view's ``mujoco.ctrl`` columns.

        A tendon no actuator transmits to carries column ``-1`` and is skipped, so commanding one
        is a no-op rather than a stray write; :meth:`__init__` already named it.

        Args:
            control: Newton control carrying ``mujoco.ctrl``.
        """
        articulation = self._articulation
        position_target = articulation.data._fixed_tendon_position_target
        commands = self._commands
        commands.zero_()
        wp.launch(
            articulation_kernels.scatter_fixed_tendon_position_targets,
            dim=position_target.shape,
            inputs=[position_target, self._actuator_columns],
            outputs=[commands],
            device=position_target.device,
        )
        self._view.set_attribute("mujoco.ctrl", control, commands)


def resolve_fixed_tendon_actuator_columns(view: ArticulationView, model: Model) -> np.ndarray | None:
    """Map each fixed tendon to the ``mujoco:actuator`` column that transmits to it.

    Args:
        view: Articulation view the actuator attributes are read through.
        model: Newton model carrying the ``mujoco:actuator`` and ``mujoco:tendon`` frequencies.

    Returns:
        Actuator column per fixed tendon, ``-1`` where no actuator transmits to that tendon, or
        None when the model carries no tendon actuators at all.
    """
    # Counts come from the view, not the model: the view's are per-articulation, while the model's
    # are scene-wide and would oversize the mapping when several articulation types coexist.
    actuator_count = view.custom_frequency_counts.get("mujoco:actuator", 0)
    tendon_count = view.custom_frequency_counts.get("mujoco:tendon", 0)
    if actuator_count == 0 or tendon_count == 0:
        return None

    # trntype says what an actuator transmits to, and is per-actuator and identical across
    # instances, so one instance's row describes the articulation. Its companion trnid names the
    # target, but the USD importer never writes it for a tendon: it resolves the row from
    # ``actuator_target_label`` when it builds the MuJoCo spec and does not persist the result, so
    # the attribute stays at its ``-1`` default. The tendon actuators are therefore matched to
    # tendons by declaration order, which both come from the same stage traversal.
    trntype = view.get_attribute("mujoco.actuator_trntype", model).numpy()[0, 0]
    tendon_actuators = np.flatnonzero(trntype == int(SolverMuJoCo.TrnType.TENDON))
    if tendon_actuators.size == 0:
        return None
    if tendon_actuators.size != tendon_count:
        raise ValueError(
            f"The articulation declares {tendon_count} fixed tendons but"
            f" {tendon_actuators.size} actuators transmit to a tendon. Ordering cannot pair them;"
            " author exactly one tendon actuator per fixed tendon."
        )

    return tendon_actuators.astype(np.int32)
