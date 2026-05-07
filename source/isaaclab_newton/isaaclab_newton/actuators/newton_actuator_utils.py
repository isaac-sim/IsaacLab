# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Newton-native actuators in Isaac Lab.

This module is organised into four sections:

1. **Warp kernels** — low-level device kernels for zeroing, masking,
   and scattering per-DOF data used by the adapter.
2. **PhysX stepping helper** — :class:`PhysxActuatorWrapper`, a
   duck-typed wrapper that exposes flat Warp arrays as the
   ``sim_state`` / ``sim_control`` protocol expected by
   :meth:`Actuator.step` on the PhysX backend.
3. **Adapter** — :class:`NewtonActuatorAdapter` manages actuator
   creation, DOF-to-actuator mapping, config overrides, stepping,
   reset, and gain reading for domain randomisation.  Used identically
   by both Newton and PhysX backends.
4. **USD authoring** — :func:`author_newton_actuator_prims` creates
   ``NewtonActuator`` USD prims from IsaacLab actuator configs so that
   both the Newton ``ModelBuilder`` (during ``add_usd``) and the PhysX
   adapter (via :meth:`NewtonActuatorAdapter.from_usd`) can construct
   :class:`Actuator` objects with correct parameters.

Why :class:`PhysxActuatorWrapper` exists only for PhysX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Newton's :meth:`Actuator.step` requires a ``sim_state`` / ``sim_control``
pair that exposes flat 1-D Warp arrays (``joint_q``, ``joint_qd``,
``joint_target_pos``, ``joint_f``, …).  On the **Newton backend** these
are the ``State`` and ``Control`` objects that the solver already owns —
no wrapper is needed because:

* The solver manages double-buffered ``State`` objects for CUDA-graph
  capture, and actuators are stepped inside the solver's own simulation
  loop where states are already available.
* Wrapping them would add indirection with no benefit; the Newton
  articulation code that calls :meth:`Actuator.step` lives in
  ``newton_manager.py`` and has direct access to the model's state.

On the **PhysX backend**, no Newton solver exists — the actuators are
stepped manually from the Lab articulation's ``write_data_to_sim``
path.  Isaac Lab stores joint data as 2-D tensors (``num_envs ×
num_joints``), so :class:`PhysxActuatorWrapper` provides zero-copy flat
views that satisfy the protocol without allocating new memory.  The
PhysX articulation code that calls :meth:`Actuator.step` lives in
``isaaclab_physx/.../articulation.py``, completely separate from the
Newton solver path, so sharing a single wrapper type across both
backends would not reduce code — it would only add coupling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp

logger = logging.getLogger(__name__)

from newton.actuators import (
    Actuator,
    Clamping,
    Delay,
)

from isaaclab.actuators import ActuatorBase, ImplicitActuator


# ===========================================================================
# 1. Warp kernels
# ===========================================================================


@wp.kernel(enable_backward=False)
def _zero_at_indices_kernel(data: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    data[indices[i]] = 0.0


@wp.kernel(enable_backward=False)
def _set_mask_kernel(mask: wp.array(dtype=wp.bool), indices: wp.array(dtype=wp.int32)):
    i = wp.tid()
    mask[indices[i]] = True


@wp.kernel(enable_backward=False)
def _scatter_gain_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.uint32),
    dof_offset: int,
    num_joints: int,
):
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    local_dof = global_dof % num_joints
    dst[env * num_joints + local_dof] = src[i]


@wp.kernel(enable_backward=False)
def _gather_gain_kernel(
    flat_src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.uint32),
    env_mask: wp.array(dtype=wp.bool),
    dof_offset: int,
    num_joints: int,
):
    """Gather from flat ``(num_envs * num_joints)`` layout into a per-actuator
    controller array, only for envs where ``env_mask`` is ``True``."""
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    if env_mask[env]:
        local_dof = global_dof % num_joints
        dst[i] = flat_src[env * num_joints + local_dof]


@wp.kernel(enable_backward=False)
def _scatter_gain_at_envs_kernel(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Scatter ``in_data[i, j]`` into ``out_data[env_ids[i], j]`` for all (i, j)."""
    i, j = wp.tid()
    out_data[env_ids[i], j] = in_data[i, j]


@wp.kernel(enable_backward=False)
def _fill_gain_at_envs_kernel(
    value: float,
    env_ids: wp.array(dtype=wp.int32),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Set ``out_data[env_ids[i], j] = value`` for all (i, j)."""
    i, j = wp.tid()
    out_data[env_ids[i], j] = value


# ===========================================================================
# 2. PhysX stepping helper
# ===========================================================================


@dataclass
class PhysxActuatorWrapper:
    """Flat-array wrapper serving as ``sim_state`` / ``sim_control`` for
    :meth:`Actuator.step` on the PhysX backend.

    Most attributes are reassigned each frame to zero-copy flat views of
    Isaac Lab's 2-D buffers. ``joint_f_2d`` is the only persistent
    allocation, sized via :meth:`create`; ``joint_f`` is its flat alias
    consumed by the Newton actuator step.
    """

    joint_q: wp.array | None = None
    joint_qd: wp.array | None = None
    joint_target_pos: wp.array | None = None
    joint_target_vel: wp.array | None = None
    joint_act: wp.array | None = None
    joint_f: wp.array | None = None
    joint_f_2d: wp.array | None = None

    @classmethod
    def create(cls, num_envs: int, num_joints: int, device: str) -> PhysxActuatorWrapper:
        """Allocate the persistent ``joint_f`` buffer for the given articulation shape."""
        w = cls()
        w.joint_f_2d = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
        w.joint_f = w.joint_f_2d.reshape(-1)
        return w


def build_actuator_telemetry(
    actuators: dict[str, ActuatorBase],
    num_envs: int,
    num_joints: int,
    device: str,
) -> tuple[wp.array, wp.array, wp.array]:
    """Build per-DOF telemetry tables.

    Per-DOF ``modes`` is ``1`` for joints covered by an
    :class:`~isaaclab.actuators.ImplicitActuator` group (shadow-PD) and
    ``0`` otherwise (copy from the simulator's actuator output).
    ``effort_limit`` carries the implicit-clip absolute limit (``inf``
    elsewhere).

    Returns:
        ``(indices, modes, effort_limit)`` Warp arrays.
    """
    modes = torch.zeros(num_joints, dtype=torch.int32, device=device)
    effort_limit = torch.full(
        (num_envs, num_joints), float("inf"), device=device, dtype=torch.float32
    )
    for actuator in actuators.values():
        if not isinstance(actuator, ImplicitActuator):
            continue
        j_ids = actuator.joint_indices
        if j_ids == slice(None) or j_ids is None:
            modes[:] = 1
            effort_limit[:] = actuator.effort_limit
        else:
            modes[j_ids.long()] = 1
            effort_limit[:, j_ids.long()] = actuator.effort_limit

    indices = wp.from_torch(
        torch.arange(num_joints, dtype=torch.int32, device=device), dtype=wp.int32
    )
    return indices, wp.from_torch(modes, dtype=wp.int32), wp.from_torch(effort_limit, dtype=wp.float32)


# ===========================================================================
# 3. Adapter — creation, config overrides, stepping, domain randomisation
# ===========================================================================


class NewtonActuatorAdapter:
    """Manages Newton-native actuators for both Newton and PhysX backends.

    Handles actuator creation (from USD or an existing list),
    DOF-to-actuator mapping, stepping, reset, and gain reading for
    domain randomisation.

    Construction:

    * **Newton backend** — pass actuators from the Newton model directly::

          adapter = NewtonActuatorAdapter(model.actuators, num_envs,
                                          num_joints, dof_offset, device)

    * **PhysX backend** — create actuators from USD prims::

          adapter = NewtonActuatorAdapter.from_usd(stage, joint_names,
                                                    num_envs, num_joints,
                                                    device)

    Then finalise::

        adapter.finalize()

    After :meth:`finalize`, the adapter exposes ``.stiffness``,
    ``.damping``, and ``.joint_indices`` for ``randomize_actuator_gains``.
    """

    def __init__(
        self,
        actuators: list[Actuator],
        num_envs: int,
        num_joints: int,
        dof_offset: int,
        device: str,
    ):
        self.actuators = actuators
        self.num_joints = num_joints

        self._num_envs = num_envs
        self._dof_offset = dof_offset
        self._device = device
        self._dof_to_actuator = self._build_dof_map()

        managed = [i for i, act_idx in enumerate(self._dof_to_actuator) if act_idx >= 0]
        if len(managed) == num_joints:
            self.joint_indices: torch.Tensor | slice = slice(None)
        else:
            self.joint_indices = torch.tensor(managed, dtype=torch.int32, device=device)

        self._states_a = [act.state() for act in actuators]
        self._states_b = [act.state() for act in actuators]

        self.stiffness: torch.Tensor | None = None
        self.damping: torch.Tensor | None = None

    # -- construction (PhysX path) -------------------------------------------

    @classmethod
    def from_usd(
        cls,
        stage: Any,
        joint_names: list[str],
        num_envs: int,
        num_joints: int,
        device: str,
        articulation_prim_path: str | None = None,
    ) -> NewtonActuatorAdapter:
        """Create an adapter by parsing ``NewtonActuator`` prims from USD.

        This is the PhysX-backend counterpart of what Newton's
        ``ModelBuilder.add_usd`` does for the Newton backend.  Both paths
        read the same ``NewtonActuator`` USD prims (authored by
        :func:`author_newton_actuator_prims`) and construct
        :class:`~newton.actuators.Actuator` objects with matching
        controllers, clampings, and delays.

        The key difference is that PhysX uses a **flat per-DOF layout**
        where joint position coordinates and velocity DOFs always have the
        same count and ordering — there are no free joints or ball joints
        that cause coordinate/DOF count divergence.  Therefore a single
        ``indices`` array is used for all index roles (``indices``,
        ``pos_indices``, ``target_pos_indices``), unlike the Newton
        builder which computes separate ``pos_indices`` from
        ``joint_q_start`` and separate ``target_pos_indices`` from
        ``joint_qd_start`` to handle floating-base articulations.

        Joints whose prims resolve to the same controller type, gains,
        clamping chain, and delay configuration are merged into a single
        :class:`Actuator` with combined index arrays, mirroring the
        grouping the Newton builder performs internally.

        Args:
            stage: The USD stage containing ``NewtonActuator`` prims.
            joint_names: All joint names in the articulation.
            num_envs: Number of environments.
            num_joints: Joints per environment in the articulation.
            device: Warp device string (e.g. ``"cuda:0"``).
            articulation_prim_path: Root prim path of the first
                environment's articulation (e.g. ``"/World/Env_0/Robot"``).
                When provided, only ``NewtonActuator`` prims under this
                subtree are considered — matching the scoped traversal
                that Newton's ``ModelBuilder.add_usd`` performs.  When
                ``None``, the entire stage is scanned (legacy behaviour).

        Returns:
            A fully constructed adapter ready for :meth:`finalize`.
        """
        actuators = cls._create_actuators_from_usd(
            stage, joint_names, num_envs, num_joints, device,
            articulation_prim_path=articulation_prim_path,
        )
        return cls(actuators, num_envs, num_joints, dof_offset=0, device=device)

    # -- public API ----------------------------------------------------------

    def finalize(self) -> None:
        """Read actuator gains and store as PyTorch tensors for DR."""
        wp_device = wp.get_device(self._device)
        flat_stiffness = wp.zeros(self._num_envs * self.num_joints, dtype=wp.float32, device=wp_device)
        flat_damping = wp.zeros(self._num_envs * self.num_joints, dtype=wp.float32, device=wp_device)

        for act in self.actuators:
            ctrl = act.controller
            if hasattr(ctrl, "kp"):
                wp.launch(
                    _scatter_gain_kernel, dim=act.indices.shape[0],
                    inputs=[ctrl.kp, flat_stiffness, act.indices, self._dof_offset, self.num_joints],
                    device=wp_device,
                )
            if hasattr(ctrl, "kd"):
                wp.launch(
                    _scatter_gain_kernel, dim=act.indices.shape[0],
                    inputs=[ctrl.kd, flat_damping, act.indices, self._dof_offset, self.num_joints],
                    device=wp_device,
                )

        self.stiffness = wp.to_torch(flat_stiffness.reshape((self._num_envs, self.num_joints)))
        self.damping = wp.to_torch(flat_damping.reshape((self._num_envs, self.num_joints)))

    def step(self, sim_state: Any, sim_control: Any, dt: float) -> None:
        """Zero actuated DOFs, step all actuators, and swap state buffers.

        Args:
            sim_state: Object with ``joint_q``, ``joint_qd``, etc.
                Newton ``State`` on the Newton backend,
                :class:`PhysxActuatorWrapper` on the PhysX backend.
            sim_control: Object with ``joint_f``, ``joint_target_pos``, etc.
                Newton ``Control`` on the Newton backend,
                :class:`PhysxActuatorWrapper` on the PhysX backend.
            dt: Physics timestep [s].
        """
        for act in self.actuators:
            wp.launch(
                _zero_at_indices_kernel,
                dim=act.indices.shape[0],
                inputs=[sim_control.joint_f, act.indices],
            )
        for act, sa, sb in zip(self.actuators, self._states_a, self._states_b):
            act.step(sim_state, sim_control, sa, sb, dt=dt)
        self._states_a, self._states_b = self._states_b, self._states_a

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset actuator states for the given environments.

        Args:
            env_ids: Environment indices to reset.  ``None`` or
                ``slice(None)`` resets all environments.
        """
        if env_ids is None or env_ids == slice(None):
            mask = None
        else:
            mask = wp.zeros(self._num_envs, dtype=wp.bool, device=self._device)
            import torch  # noqa: PLC0415
            if isinstance(env_ids, torch.Tensor):
                idx = wp.from_torch(env_ids.to(device=self._device).contiguous().to(torch.int32), dtype=wp.int32)
            else:
                idx = wp.array(list(env_ids), dtype=wp.int32, device=self._device)
            wp.launch(_set_mask_kernel, dim=len(idx), inputs=[mask, idx], device=self._device)

        for sa, sb in zip(self._states_a, self._states_b):
            if sa is not None:
                sa.reset(mask)
            if sb is not None:
                sb.reset(mask)

    @property
    def is_all_graphable(self) -> bool:
        """``True`` when all actuators are CUDA-graph-safe."""
        return len(self.actuators) > 0 and all(a.is_graphable() for a in self.actuators)

    def update_gain_at_env_ids(
        self,
        gain: str,
        values: torch.Tensor | wp.array | float,
        env_ids: wp.array,
    ) -> wp.array:
        """Scatter ``values`` into :attr:`stiffness` or :attr:`damping` at *env_ids*.

        Shared backend-independent step used by ``write_actuator_*_to_sim`` to
        keep the kp/kd buffer the adapter exposes for DR in sync with what
        each Newton actuator's controller will receive.

        Args:
            gain: ``"stiffness"`` or ``"damping"``.
            values: Per-env-per-joint values shape ``(len(env_ids), num_joints)``,
                or a scalar to broadcast to every (env, joint).
            env_ids: Warp int32 array of env indices to update.

        Returns:
            Warp view of the updated ``(num_envs, num_joints)`` buffer.
        """
        if gain == "stiffness":
            buf = self.stiffness
        elif gain == "damping":
            buf = self.damping
        else:
            raise ValueError(f"gain must be 'stiffness' or 'damping', got {gain!r}")
        buf_wp = wp.from_torch(buf, dtype=wp.float32)
        if isinstance(values, float):
            wp.launch(
                _fill_gain_at_envs_kernel,
                dim=(env_ids.shape[0], self.num_joints),
                inputs=[values, env_ids],
                outputs=[buf_wp],
                device=self._device,
            )
        else:
            wp.launch(
                _scatter_gain_at_envs_kernel,
                dim=(env_ids.shape[0], self.num_joints),
                inputs=[values, env_ids],
                outputs=[buf_wp],
                device=self._device,
            )
        return buf_wp

    def write_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | wp.array | float,
        env_ids: wp.array,
        env_mask: wp.array,
        propagate_fn: Callable[["NewtonActuatorAdapter", Actuator, Any, str, wp.array, wp.array], None],
    ) -> None:
        """Update the kp buffer at *env_ids* and push the new values into each Newton controller.

        The per-actuator propagation step is backend-specific (Newton uses
        :meth:`ArticulationView.set_actuator_parameter`; PhysX scatters via a
        local Warp kernel), so the caller injects it as *propagate_fn*.
        """
        self._write_gain_to_sim("stiffness", "kp", stiffness, env_ids, env_mask, propagate_fn)

    def write_damping_to_sim(
        self,
        damping: torch.Tensor | wp.array | float,
        env_ids: wp.array,
        env_mask: wp.array,
        propagate_fn: Callable[["NewtonActuatorAdapter", Actuator, Any, str, wp.array, wp.array], None],
    ) -> None:
        """Update the kd buffer at *env_ids* and push the new values into each Newton controller."""
        self._write_gain_to_sim("damping", "kd", damping, env_ids, env_mask, propagate_fn)

    def _write_gain_to_sim(
        self,
        gain: str,
        attr: str,
        values: torch.Tensor | wp.array | float,
        env_ids: wp.array,
        env_mask: wp.array,
        propagate_fn: Callable[["NewtonActuatorAdapter", Actuator, Any, str, wp.array, wp.array], None],
    ) -> None:
        """Shared body for :meth:`write_stiffness_to_sim` / :meth:`write_damping_to_sim`."""
        buf = self.update_gain_at_env_ids(gain, values, env_ids)
        for newton_act in self.actuators:
            ctrl = newton_act.controller
            if hasattr(ctrl, attr):
                propagate_fn(self, newton_act, ctrl, attr, buf, env_mask)

    # -- config helpers (used by both adapter and authoring) -----------------

    @staticmethod
    def _resolve_per_dof(
        value: dict[str, float | int] | float | int | None,
        joint_names: list[str],
        cast: type = float,
    ) -> dict[str, float | int]:
        """Expand a scalar or dict config value into a per-DOF dict.

        When *value* is a dict, keys are treated as regex patterns and
        matched against *joint_names* via :func:`re.fullmatch`.
        """
        import re  # noqa: PLC0415

        if value is None:
            return {}
        if isinstance(value, (int, float)):
            return {name: cast(value) for name in joint_names}
        if isinstance(value, dict):
            result: dict[str, float | int] = {}
            for name in joint_names:
                for pattern, v in value.items():
                    if re.fullmatch(pattern, name):
                        result[name] = cast(v)
                        break
            return result
        return {}

    # -- private helpers -----------------------------------------------------

    def _build_dof_map(self) -> list[int]:
        """Build a per-DOF lookup: local DOF index -> actuator list index."""
        dof_to_actuator: list[int] = [-1] * self.num_joints

        for act_idx, act in enumerate(self.actuators):
            all_indices = act.indices.numpy()
            num_per_act = len(all_indices) // self._num_envs
            env0_indices = all_indices[:num_per_act]
            for global_dof in env0_indices:
                local_dof = global_dof - self._dof_offset
                if 0 <= local_dof < self.num_joints:
                    dof_to_actuator[local_dof] = act_idx

        return dof_to_actuator

    @staticmethod
    def _actuator_signature(parsed: Any) -> tuple:
        """Build a hashable key from a parsed actuator spec for grouping.

        Joints whose prims resolve to the same signature share identical
        controller type, gains, clamping chain, and delay configuration and
        can therefore be merged into a single :class:`Actuator` with combined
        index arrays.
        """
        ctrl_resolved = parsed.controller_class.resolve_arguments(
            dict(parsed.controller_kwargs),
        )
        ctrl_key = (parsed.controller_class, tuple(sorted(ctrl_resolved.items())))

        comp_keys: list[tuple] = []
        for comp_cls, comp_kwargs in parsed.component_specs:
            resolved = comp_cls.resolve_arguments(comp_kwargs)
            comp_keys.append((comp_cls, tuple(sorted(resolved.items()))))
        comp_keys.sort(key=lambda t: t[0].__name__)

        return (ctrl_key, tuple(comp_keys))

    @staticmethod
    def _create_actuators_from_usd(
        stage: Any,
        joint_names: list[str],
        num_envs: int,
        num_total_joints: int,
        device: str,
        articulation_prim_path: str | None = None,
    ) -> list[Actuator]:
        """Parse ``NewtonActuator`` prims and instantiate standalone actuators.

        This mirrors the actuator construction that Newton's
        ``ModelBuilder.add_usd`` performs, but operates independently of a
        Newton ``Model``.  It is used on the PhysX backend where there is no
        Newton simulation — actuators are stepped manually via the adapter.

        Because PhysX articulations have no free or ball joints, every
        joint's coordinate count equals its DOF count.  A single
        ``indices`` array is therefore sufficient for all index roles
        (``indices``, ``pos_indices``, ``target_pos_indices``).

        Joints with identical controller type, gains, clamping chain, and
        delay are merged into one :class:`Actuator` with combined indices.

        Each per-DOF scalar parameter (``kp``, ``kd``, ``saturation_effort``,
        etc.) is broadcast via :func:`wp.full` to match the group size.
        Parameters marked as ``SHARED_PARAMS`` on the controller or clamping
        class (e.g. ``model_path``, ``lookup_positions``) are passed through
        directly without broadcast.
        """
        from collections import defaultdict  # noqa: PLC0415

        from newton.actuators import parse_actuator_prim  # noqa: PLC0415
        from pxr import Usd  # noqa: PLC0415

        wp_device = wp.get_device(device)

        joint_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(joint_names)}

        if articulation_prim_path is not None:
            root_prim = stage.GetPrimAtPath(articulation_prim_path)
        else:
            root_prim = stage.GetPseudoRoot()

        parsed_per_joint: dict[int, Any] = {}
        for prim in Usd.PrimRange(root_prim):
            parsed = parse_actuator_prim(prim)
            if parsed is None:
                continue
            target_name = parsed.target_path.rsplit("/", 1)[-1]
            if target_name in joint_name_to_idx:
                parsed_per_joint[joint_name_to_idx[target_name]] = parsed

        if not parsed_per_joint:
            raise ValueError(
                f"No NewtonActuator prims found targeting any of: {joint_names}"
            )

        groups: dict[tuple, list[int]] = defaultdict(list)
        sig_to_parsed: dict[tuple, Any] = {}
        for local_idx, parsed in sorted(parsed_per_joint.items()):
            sig = NewtonActuatorAdapter._actuator_signature(parsed)
            groups[sig].append(local_idx)
            if sig not in sig_to_parsed:
                sig_to_parsed[sig] = parsed

        actuators = []
        for sig, local_indices in groups.items():
            parsed = sig_to_parsed[sig]

            flat_indices = np.array(
                [idx + e * num_total_joints for e in range(num_envs) for idx in local_indices],
                dtype=np.uint32,
            )
            indices = wp.array(flat_indices, device=wp_device)
            num_dofs_in_group = len(local_indices) * num_envs

            # Controller
            ctrl_kwargs = dict(parsed.controller_kwargs)
            resolved = parsed.controller_class.resolve_arguments(ctrl_kwargs)
            shared_ctrl = getattr(parsed.controller_class, "SHARED_PARAMS", set())
            ctrl_arrays = {}
            for key, val in resolved.items():
                if key in shared_ctrl:
                    ctrl_arrays[key] = val
                else:
                    ctrl_arrays[key] = wp.full(num_dofs_in_group, float(val), dtype=wp.float32, device=wp_device)
            controller = parsed.controller_class(**ctrl_arrays)

            # Components (delay + clampings)
            clampings = []
            delay = None
            for comp_cls, comp_kwargs in parsed.component_specs:
                if issubclass(comp_cls, Delay):
                    resolved_kw = Delay.resolve_arguments(comp_kwargs)
                    delay_steps = int(resolved_kw.get("delay_steps", 0))
                    if delay_steps > 0:
                        delay_arr = wp.full(num_dofs_in_group, delay_steps, dtype=wp.int32, device=wp_device)
                        delay = Delay(delay_steps=delay_arr, max_delay=delay_steps)
                elif issubclass(comp_cls, Clamping):
                    resolved_kw = comp_cls.resolve_arguments(comp_kwargs)
                    shared_clamp = getattr(comp_cls, "SHARED_PARAMS", set())
                    clamp_arrays = {}
                    for k, v in resolved_kw.items():
                        if k in shared_clamp:
                            clamp_arrays[k] = v
                        else:
                            clamp_arrays[k] = wp.full(
                                num_dofs_in_group, float(v), dtype=wp.float32, device=wp_device,
                            )
                    clampings.append(comp_cls(**clamp_arrays))

            actuator = Actuator(
                indices=indices,
                controller=controller,
                delay=delay,
                clamping=clampings if clampings else None,
            )
            actuators.append(actuator)

        return actuators


# ===========================================================================
# 4. USD authoring — create NewtonActuator prims from Lab actuator configs
# ===========================================================================


def author_newton_actuator_prims(
    stage: Any,
    articulation_prim_path: str,
    actuator_cfgs: dict[str, Any],
) -> None:
    """Author ``NewtonActuator`` USD prims from IsaacLab actuator configs.

    For every joint covered by an explicit (non-implicit) Lab actuator config,
    any existing ``NewtonActuator`` prim targeting that joint is replaced by a
    new one created from the config values.  Joints **not** covered by any
    Lab config keep their USD-authored actuators unchanged.

    The supported config-to-schema mapping is:

    * :class:`~isaaclab.actuators.IdealPDActuatorCfg` ->
      ``NewtonPDControlAPI`` + ``NewtonMaxEffortClampingAPI``
    * :class:`~isaaclab.actuators.DCMotorCfg` ->
      ``NewtonPDControlAPI`` + ``NewtonDCMotorClampingAPI``
    * :class:`~isaaclab.actuators.DelayedPDActuatorCfg` ->
      same as ``IdealPDActuatorCfg`` + ``NewtonActuatorDelayAPI``
    * :class:`~isaaclab.actuators.RemotizedPDActuatorCfg` ->
      same as ``DelayedPDActuatorCfg`` + ``NewtonPositionBasedClampingAPI``
    * :class:`~isaaclab.actuators.ActuatorNetMLPCfg` /
      :class:`~isaaclab.actuators.ActuatorNetLSTMCfg` ->
      ``NewtonNeuralControlAPI`` (+ ``NewtonDCMotorClampingAPI``)

    Must be called **after** the articulation is spawned (joint prims exist
    on stage) and **before** the cloner / ``ModelBuilder.add_usd`` reads
    the stage.

    Args:
        stage: The USD stage to author prims on.
        articulation_prim_path: Root prim path of the articulation
            (e.g. ``"/World/Env_0/Robot"``).  Must not contain wildcards.
        actuator_cfgs: Mapping of group name to ``ActuatorBaseCfg``.
    """
    from pxr import Sdf  # noqa: PLC0415

    from isaaclab.actuators import ImplicitActuator  # noqa: PLC0415
    from isaaclab.utils.string import resolve_matching_names  # noqa: PLC0415

    art_prim = stage.GetPrimAtPath(articulation_prim_path)
    if not art_prim.IsValid():
        raise ValueError(f"Articulation prim not found: {articulation_prim_path}")

    joint_inventory = _collect_joint_prims(art_prim)
    all_joint_names = list(joint_inventory.keys())

    covered_joint_paths: set[str] = set()
    resolve = NewtonActuatorAdapter._resolve_per_dof

    cfg_entries: list[tuple[str, Any, list[str]]] = []
    for group_name, cfg in actuator_cfgs.items():
        cls_type = cfg.class_type
        is_implicit = (
            "ImplicitActuator" in cls_type
            if isinstance(cls_type, str)
            else issubclass(cls_type, ImplicitActuator)
        )
        if is_implicit:
            continue

        _ids, joint_names = resolve_matching_names(cfg.joint_names_expr, all_joint_names)
        if not joint_names:
            continue

        cfg_entries.append((group_name, cfg, joint_names))
        for jname in joint_names:
            covered_joint_paths.add(joint_inventory[jname])

    _remove_actuator_prims_for_joints(art_prim, covered_joint_paths)

    from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg  # noqa: PLC0415
    from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg  # noqa: PLC0415
    from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg, RemotizedPDActuatorCfg  # noqa: PLC0415

    _SUPPORTED_CFG_TYPES = (
        IdealPDActuatorCfg,
        DCMotorCfg,
        DelayedPDActuatorCfg,
        RemotizedPDActuatorCfg,
        ActuatorNetMLPCfg,
        ActuatorNetLSTMCfg,
    )

    for group_name, cfg, joint_names in cfg_entries:
        if not isinstance(cfg, _SUPPORTED_CFG_TYPES):
            logger.warning(
                "Actuator group '%s' uses config type '%s' which is not supported by Newton-native"
                " actuator authoring. The group will be skipped.",
                group_name,
                type(cfg).__name__,
            )
            continue
        stiffness_map = resolve(getattr(cfg, "stiffness", None), joint_names)
        damping_map = resolve(getattr(cfg, "damping", None), joint_names)
        effort_map = resolve(getattr(cfg, "effort_limit", None), joint_names)

        is_neural = isinstance(cfg, (ActuatorNetMLPCfg, ActuatorNetLSTMCfg))
        is_remotized = isinstance(cfg, RemotizedPDActuatorCfg)
        is_dc_motor = isinstance(cfg, DCMotorCfg)
        is_delayed = isinstance(cfg, DelayedPDActuatorCfg)

        vel_limit_map = resolve(getattr(cfg, "velocity_limit", None), joint_names) if is_dc_motor else {}
        sat_effort_map = resolve(getattr(cfg, "saturation_effort", None), joint_names) if is_dc_motor else {}

        raw_delay = getattr(cfg, "max_delay", 0) if is_delayed else 0
        delay_map = resolve(raw_delay, joint_names, cast=int) if raw_delay else {}

        patched_model_path: str | None = None
        if is_neural:
            meta: dict[str, Any] = {}
            if isinstance(cfg, ActuatorNetMLPCfg):
                meta["model_type"] = "mlp"
                meta["input_order"] = cfg.input_order
                meta["input_idx"] = list(cfg.input_idx)
                meta["pos_scale"] = cfg.pos_scale
                meta["vel_scale"] = cfg.vel_scale
                meta["torque_scale"] = cfg.torque_scale
            else:
                meta["model_type"] = "lstm"
            patched_model_path = _resave_checkpoint_with_metadata(cfg.network_file, meta)

        for jname in joint_names:
            joint_prim_path = joint_inventory[jname]

            schemas: list[str] = []
            attrs: dict[str, float | int] = {}
            array_attrs: dict[str, list[float]] = {}

            if is_neural:
                schemas.append("NewtonNeuralControlAPI")
            else:
                schemas.append("NewtonPDControlAPI")
                attrs["kp"] = stiffness_map.get(jname, 0.0)
                attrs["kd"] = damping_map.get(jname, 0.0)

            if is_dc_motor:
                schemas.append("NewtonDCMotorClampingAPI")
                attrs["saturation_effort"] = sat_effort_map.get(jname, 0.0)
                if jname in vel_limit_map:
                    attrs["velocity_limit"] = vel_limit_map[jname]
                if jname in effort_map:
                    attrs["max_motor_effort"] = effort_map[jname]
            elif jname in effort_map:
                schemas.append("NewtonMaxEffortClampingAPI")
                attrs["max_effort"] = effort_map[jname]

            if is_remotized and isinstance(cfg, RemotizedPDActuatorCfg):
                lookup = cfg.joint_parameter_lookup
                schemas.append("NewtonPositionBasedClampingAPI")
                array_attrs["lookup_positions"] = [row[0] for row in lookup]
                array_attrs["lookup_efforts"] = [row[2] for row in lookup]

            delay_steps = delay_map.get(jname, 0)
            if delay_steps > 0:
                schemas.append("NewtonActuatorDelayAPI")
                attrs["delay_steps"] = delay_steps
                attrs["max_delay"] = delay_steps

            act_prim_path = f"{articulation_prim_path}/{group_name}_{jname}_actuator"
            act_prim = stage.DefinePrim(act_prim_path, "NewtonActuator")

            existing = act_prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
            existing.prependedItems = list(schemas)
            act_prim.SetMetadata("apiSchemas", existing)

            rel = act_prim.CreateRelationship("newton:targets")
            rel.SetTargets([Sdf.Path(joint_prim_path)])

            if patched_model_path is not None:
                act_prim.CreateAttribute("newton:modelPath", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(patched_model_path)
                )

            for attr_name, attr_val in attrs.items():
                usd_name = f"newton:{_snake_to_camel(attr_name)}"
                if isinstance(attr_val, int):
                    act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Int).Set(attr_val)
                else:
                    act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Float).Set(float(attr_val))

            for attr_name, attr_val in array_attrs.items():
                usd_name = f"newton:{_snake_to_camel(attr_name)}"
                act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.FloatArray).Set(attr_val)


# ---------------------------------------------------------------------------
# USD authoring — private helpers
# ---------------------------------------------------------------------------

_SNAKE_TO_CAMEL_RE = __import__("re").compile(r"_([a-z])")


def _snake_to_camel(name: str) -> str:
    """Convert a snake_case name to camelCase."""
    return _SNAKE_TO_CAMEL_RE.sub(lambda m: m.group(1).upper(), name)


def _collect_joint_prims(art_prim: Any) -> dict[str, str]:
    """Collect all joint prims under an articulation subtree.

    Returns:
        Ordered mapping of joint name to full prim path.
    """
    from pxr import Usd  # noqa: PLC0415

    _JOINT_TYPES = {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}

    joints: dict[str, str] = {}
    for prim in Usd.PrimRange(art_prim):
        if prim.GetTypeName() in _JOINT_TYPES:
            joints[prim.GetName()] = str(prim.GetPath())
    return joints


def _remove_actuator_prims_for_joints(
    art_prim: Any,
    joint_paths: set[str],
) -> None:
    """Deactivate ``NewtonActuator`` prims whose target is in *joint_paths*.

    Deactivated prims are invisible to ``Usd.PrimRange`` and therefore
    ignored by ``ModelBuilder.add_usd``.  Using ``SetActive(False)``
    instead of ``RemovePrim`` works correctly when the prim originates
    from a USD reference or payload.

    Only prims under the *art_prim* subtree are considered.
    """
    from pxr import Usd  # noqa: PLC0415

    to_deactivate: list = []
    for prim in Usd.PrimRange(art_prim):
        if prim.GetTypeName() != "NewtonActuator":
            continue
        rel = prim.GetRelationship("newton:targets")
        if rel and rel.IsValid():
            for target in rel.GetTargets():
                if str(target) in joint_paths:
                    to_deactivate.append(prim)
                    break

    for prim in to_deactivate:
        prim.SetActive(False)


def _resave_checkpoint_with_metadata(
    original_path: str,
    metadata: dict[str, Any],
) -> str:
    """Re-save a neural-network checkpoint with updated metadata.

    Loads the original TorchScript or dict checkpoint, merges *metadata*
    into any existing metadata (Lab config values take precedence), and
    writes the result to a temporary ``.pt`` file that persists for the
    lifetime of the process.

    Returns:
        Path to the temporary checkpoint file.
    """
    import json
    import tempfile

    import torch

    extra_files: dict[str, str] = {"metadata.json": ""}
    is_torchscript = True
    try:
        net = torch.jit.load(original_path, map_location="cpu", _extra_files=extra_files)
        existing_meta = json.loads(extra_files["metadata.json"]) if extra_files["metadata.json"] else {}
    except Exception:
        is_torchscript = False
        checkpoint = torch.load(original_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                f"Cannot load checkpoint at '{original_path}'; "
                "expected a TorchScript archive or a dict with a 'model' key"
            )
        net = checkpoint["model"]
        existing_meta = checkpoint.get("metadata", {})

    merged = {**existing_meta, **metadata}

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    if is_torchscript:
        extra_out = {"metadata.json": json.dumps(merged)}
        torch.jit.save(net, tmp.name, _extra_files=extra_out)
    else:
        torch.save({"model": net, "metadata": merged}, tmp.name)

    return tmp.name
