# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MJWarp fixed-tendon command submission.

Tendons are a MuJoCo concept, so a fixed tendon is driven by whichever MuJoCo actuator transmits
to it. :class:`~newton.selection.ArticulationView` exposes the ``mujoco:actuator`` frequency, so
this maps each fixed tendon to its actuator column and lets the view own the world and instance
layout; it owns the per-step write that carries buffered targets into ``mujoco.ctrl``.

:class:`MjWarpTendonControl` holds the articulation it drives, so it reuses that asset's index
resolution and shape checking rather than restating them. The command buffer itself stays on
:class:`~isaaclab_newton.assets.ArticulationData`, beside the other fixed-tendon buffers.

Tendon submission is solver-specific -- ``mujoco.ctrl`` exists only under MJWarp -- so it lives
beside the MJWarp manager rather than in the backend-neutral actuator control. The articulation
owns the target buffer and asks its manager to submit; only
:class:`~isaaclab_newton.physics.NewtonMJWarpManager` imports this module.
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
from isaaclab_newton.assets.articulation import kernels as articulation_kernels

from ._mjwarp_view_compat import ensure_newton_custom_frequency_api

if TYPE_CHECKING:
    from newton import Control, Model
    from newton.selection import ArticulationView

    from isaaclab_newton.assets.articulation.articulation import Articulation

logger = logging.getLogger(__name__)


class MjWarpTendonControl:
    """Drives an articulation's fixed tendons through MuJoCo's native tendon actuators.

    Created on demand by :meth:`~isaaclab_newton.physics.NewtonMJWarpManager.fixed_tendon_control`
    when the model carries tendon actuators. Command tendons through the articulation's
    backend-neutral
    :meth:`~isaaclab.assets.articulation.BaseArticulation.set_fixed_tendon_position_target_index`
    rather than through this internal adapter.
    """

    @classmethod
    def create(cls, articulation: Articulation, model: Model) -> MjWarpTendonControl | None:
        """Build the adapter, or None when no actuator drives any of the tendons.

        Args:
            articulation: Newton articulation owning the fixed tendons.
            model: Newton model the articulation's view selects from.

        Returns:
            The adapter, or None when the articulation's tendons are all passive.
        """
        # Single compatibility seam: on Newton 1.6 this returns both unchanged. Everything below,
        # and every other module, is written against 1.6 only.
        view, model = ensure_newton_custom_frequency_api(articulation.root_view, model)
        mapping = resolve_fixed_tendon_actuator_columns(view, model)
        if mapping is None or mapping[0].size == 0:
            return None
        return cls(articulation, mapping, view)

    def __init__(self, articulation: Articulation, mapping: tuple[np.ndarray, np.ndarray], view: ArticulationView):
        """Bind a tendon-to-actuator mapping to the articulation it drives.

        Args:
            articulation: Newton articulation owning the fixed tendons.
            mapping: ``(actuator_columns, target_tendon_ids)`` from
                :func:`resolve_fixed_tendon_actuator_columns`, one entry per driving actuator.
            view: Articulation view exposing the ``mujoco:actuator`` custom frequency.
        """
        actuator_columns, target_tendon_ids = mapping
        self._articulation = articulation
        self._actuator_columns = wp.array(actuator_columns, dtype=wp.int32, device=articulation.device)
        self._target_tendon_ids = wp.array(target_tendon_ids, dtype=wp.int32, device=articulation.device)
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
        # :meth:`set_position_target_index` are a device array, so checking them there would synchronize
        # the GPU on every step.
        driven = set(target_tendon_ids.tolist())
        passive = [name for tendon, name in enumerate(articulation.fixed_tendon_names) if tendon not in driven]
        if passive:
            logger.warning(
                "Fixed tendons %s have no MuJoCo actuator, so commanding them has no effect. Author an"
                " actuator whose transmission is each tendon to drive them.",
                passive,
            )

    def set_position_target_index(
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

        Only actuators that transmit to a tendon have an entry, so commanding a tendon nothing
        drives is a no-op rather than a stray write; :meth:`__init__` already named it.

        Args:
            control: Newton control carrying ``mujoco.ctrl``.
        """
        articulation = self._articulation
        position_target = articulation.data._fixed_tendon_position_target
        commands = self._commands
        commands.zero_()
        wp.launch(
            articulation_kernels.scatter_fixed_tendon_position_targets,
            dim=(position_target.shape[0], self._actuator_columns.shape[0]),
            inputs=[position_target, self._actuator_columns, self._target_tendon_ids],
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
        ``(actuator_columns, target_tendon_ids)`` -- one entry per actuator that transmits to a
        tendon, so a tendon nothing drives has no entry and several actuators may name the same
        tendon. None when the model carries no tendon actuators at all.

    Raises:
        ValueError: If a tendon actuator targets something that is not one of the articulation's
            fixed tendons.
    """
    # Counts come from the view, not the model: the view's are per-articulation, while the model's
    # are scene-wide and would oversize the mapping when several articulation types coexist.
    actuator_count = view.custom_frequency_counts.get("mujoco:actuator", 0)
    tendon_count = view.custom_frequency_counts.get("mujoco:tendon", 0)
    if actuator_count == 0 or tendon_count == 0:
        return None

    # trntype says what an actuator transmits to and is identical across instances, so one row
    # describes the articulation. Its companion trnid is never written for a USD tendon actuator, so
    # the target is matched the way the solver matches it: by the actuator's target label against
    # the tendon names, which come from the same labels.
    trntype = view.get_attribute("mujoco.actuator_trntype", model).numpy()[0, 0]
    tendon_actuators = np.flatnonzero(trntype == int(SolverMuJoCo.TrnType.TENDON))
    if tendon_actuators.size == 0:
        return None

    # The TARGET relation, not the display label: ``custom_frequency_labels`` holds actuator names,
    # so matching tendons against it would resolve nothing. ``actuator_target_label`` is a
    # ``dtype=str`` model attribute -- the view's stride-based ``get_attribute`` cannot carry
    # strings -- so it is read off the model and indexed by this articulation's rows, the same way
    # the view builds its own label lists.
    target_labels = model.mujoco.actuator_target_label
    rows = _actuator_template_rows(view, model)
    tendon_names = list(view.tendon_names)
    actuator_columns, target_tendon_ids = [], []
    for column in tendon_actuators:
        label = str(target_labels[rows[column]])
        name = label.rsplit("/", 1)[-1]
        if name not in tendon_names:
            raise ValueError(
                f"Actuator column {column} targets '{label}', which is not one of this articulation's fixed"
                f" tendons {tendon_names}."
            )
        actuator_columns.append(int(column))
        target_tendon_ids.append(tendon_names.index(name))
    return np.array(actuator_columns, dtype=np.int32), np.array(target_tendon_ids, dtype=np.int32)


def _actuator_template_rows(view: ArticulationView, model: Model) -> np.ndarray:
    """Model-global ``mujoco:actuator`` rows owned by this view's template articulation.

    Newton states the relation in ``custom_frequency_articulation`` (newton-physics/newton#4017),
    so the rows are read rather than inferred from strides.

    Args:
        view: Articulation view the actuator columns belong to.
        model: Newton model carrying the ownership relation.

    Returns:
        The model-global actuator rows of one articulation, in column order.
    """
    owners = model.custom_frequency_articulation["mujoco:actuator"]
    template = int(_to_numpy(view.articulation_ids)[0])
    return np.flatnonzero(_to_numpy(owners) == template)


def _to_numpy(value) -> np.ndarray:
    """Return *value* as a numpy array, whether it is a Warp array already or a sequence."""
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)
