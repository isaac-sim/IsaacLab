# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-actuator adapter shared by the Newton and PhysX backends.

:class:`NewtonActuatorAdapter` manages actuator creation, DOF-to-actuator
mapping, stepping, reset, and gain reading for domain randomisation.
Used identically by both Newton and PhysX backends.

The companion helper :func:`build_implicit_dof_mask` is consumed by the
in-graph post-actuator kernel
(:func:`~isaaclab_newton.actuators.kernels.synch_torque_and_apply_implicit_feedforwards`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import warp as wp

from newton.actuators import Actuator, Clamping, Delay

from isaaclab.actuators import ActuatorBase, ImplicitActuator

from .kernels import (
    fill_gain_at_envs_kernel,
    gather_gain_kernel,
    scatter_gain_at_envs_kernel,
    scatter_gain_kernel,
    set_mask_kernel,
    zero_at_indices_kernel,
)


def _actuator_signature(parsed: Any) -> tuple:
    """Build a hashable key from a parsed actuator spec for grouping.

    Joints whose prims resolve to the same signature share identical
    controller type, gains, clamping chain, and delay configuration and
    can therefore be merged into a single :class:`~newton.actuators.Actuator`
    with combined index arrays.
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


def build_implicit_dof_mask(
    actuators: dict[str, ActuatorBase],
    num_joints: int,
    device: str,
) -> wp.array:
    """Per-DOF mask consumed by the in-graph implicit-FF kernel.

    Entry is ``1`` for DOFs covered by an
    :class:`~isaaclab.actuators.ImplicitActuator` group, ``0`` otherwise.
    """
    modes = torch.zeros(num_joints, dtype=torch.int32, device=device)
    for actuator in actuators.values():
        if not isinstance(actuator, ImplicitActuator):
            continue
        j_ids = actuator.joint_indices
        if j_ids == slice(None) or j_ids is None:
            modes[:] = 1
        else:
            modes[j_ids.long()] = 1
    return wp.from_torch(modes, dtype=wp.int32)


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

        # Per-actuator gain propagator. Configured once at adapter setup time
        # via :meth:`set_view_propagator` (Newton: simulator-side scatter API)
        # or :meth:`set_kernel_propagator` (PhysX: local Warp scatter kernel).
        # Stays ``None`` if gain DR is not used; ``write_*_to_sim`` then
        # only updates the adapter's gain buffer without pushing to controllers.
        self._propagator: Callable[[Actuator, Any, str, wp.array, wp.array], None] | None = None

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
    ) -> "NewtonActuatorAdapter":
        """Create an adapter by parsing ``NewtonActuator`` prims from USD.

        This is the PhysX-backend counterpart of what Newton's
        ``ModelBuilder.add_usd`` does for the Newton backend.  Both paths
        read the same ``NewtonActuator`` USD prims (authored by
        :func:`~isaaclab_newton.actuators.authoring.author_newton_actuator_prims`)
        and construct :class:`~newton.actuators.Actuator` objects with
        matching controllers, clampings, and delays.

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
        actuators = _create_actuators_from_usd(
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
                    scatter_gain_kernel, dim=act.indices.shape[0],
                    inputs=[ctrl.kp, flat_stiffness, act.indices, self._dof_offset, self.num_joints],
                    device=wp_device,
                )
            if hasattr(ctrl, "kd"):
                wp.launch(
                    scatter_gain_kernel, dim=act.indices.shape[0],
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
                :class:`~isaaclab_newton.actuators.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            sim_control: Object with ``joint_f``, ``joint_target_pos``, etc.
                Newton ``Control`` on the Newton backend,
                :class:`~isaaclab_newton.actuators.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            dt: Physics timestep [s].
        """
        for act in self.actuators:
            wp.launch(
                zero_at_indices_kernel,
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
                ``slice(None)`` resets all environments. A partial slice
                (e.g. ``slice(0, 5)``) is materialized to explicit indices.
        """
        if env_ids is None or env_ids == slice(None):
            mask = None
        else:
            # Normalize a partial slice to an explicit index list before
            # building the wp.array — slices aren't iterable.
            if isinstance(env_ids, slice):
                env_ids = list(range(*env_ids.indices(self._num_envs)))
            if isinstance(env_ids, torch.Tensor):
                if env_ids.numel() == 0:
                    return
                idx = wp.from_torch(env_ids.to(device=self._device).contiguous().to(torch.int32), dtype=wp.int32)
            else:
                if len(env_ids) == 0:
                    return
                idx = wp.array(list(env_ids), dtype=wp.int32, device=self._device)
            mask = wp.zeros(self._num_envs, dtype=wp.bool, device=self._device)
            wp.launch(set_mask_kernel, dim=idx.shape[0], inputs=[mask, idx], device=self._device)

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

        Shared backend-independent step used by ``write_*_to_sim`` to
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
                fill_gain_at_envs_kernel,
                dim=(env_ids.shape[0], self.num_joints),
                inputs=[values, env_ids],
                outputs=[buf_wp],
                device=self._device,
            )
        else:
            wp.launch(
                scatter_gain_at_envs_kernel,
                dim=(env_ids.shape[0], self.num_joints),
                inputs=[values, env_ids],
                outputs=[buf_wp],
                device=self._device,
            )
        return buf_wp

    def set_view_propagator(self, root_view: Any) -> None:
        """Configure gain propagation via Newton's simulator-side scatter API.

        Used on the Newton backend, where each per-actuator gain push goes
        through ``ArticulationView.set_actuator_parameter``. Any one
        articulation's view works since the call dispatches on the actuator
        object (model-scoped), not on the view's articulation.
        """
        def _push(
            actuator: Actuator, controller: Any, attr: str,
            values: wp.array, env_mask: wp.array,
        ) -> None:
            root_view.set_actuator_parameter(
                actuator=actuator, component=controller, name=attr,
                values=values, mask=env_mask,
            )
        self._propagator = _push

    def set_kernel_propagator(self) -> None:
        """Configure gain propagation via the local ``gather_gain_kernel``.

        Used on the PhysX backend, where there is no simulator-side scatter
        API. The kernel reads the adapter's per-DOF gain buffer at each
        actuator's flat indices and writes into ``controller.kp`` /
        ``controller.kd``.
        """
        def _push(
            actuator: Actuator, controller: Any, attr: str,
            values: wp.array, env_mask: wp.array,
        ) -> None:
            wp.launch(
                gather_gain_kernel,
                dim=actuator.indices.shape[0],
                inputs=[
                    values.flatten(), getattr(controller, attr), actuator.indices,
                    env_mask, self._dof_offset, self.num_joints,
                ],
                device=self._device,
            )
        self._propagator = _push

    def write_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | wp.array | float,
        env_ids: wp.array,
        env_mask: wp.array,
    ) -> None:
        """Update the kp buffer at *env_ids* and push the new values into each Newton controller."""
        self._write_gain_to_sim("stiffness", "kp", stiffness, env_ids, env_mask)

    def write_damping_to_sim(
        self,
        damping: torch.Tensor | wp.array | float,
        env_ids: wp.array,
        env_mask: wp.array,
    ) -> None:
        """Update the kd buffer at *env_ids* and push the new values into each Newton controller."""
        self._write_gain_to_sim("damping", "kd", damping, env_ids, env_mask)

    def _write_gain_to_sim(
        self,
        gain: str,
        attr: str,
        values: torch.Tensor | wp.array | float,
        env_ids: wp.array,
        env_mask: wp.array,
    ) -> None:
        """Shared body for :meth:`write_stiffness_to_sim` / :meth:`write_damping_to_sim`."""
        buf = self.update_gain_at_env_ids(gain, values, env_ids)
        if self._propagator is None:
            return
        for newton_act in self.actuators:
            ctrl = newton_act.controller
            if hasattr(ctrl, attr):
                self._propagator(newton_act, ctrl, attr, buf, env_mask)

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
        sig = _actuator_signature(parsed)
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
