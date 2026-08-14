# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-actuator adapter shared by Newton, PhysX, and OVPhysX.

Owns the actuator-state lifecycle, the pre-clamp computed-effort buffer,
and the per-step ``step`` / ``reset`` / ``finalize`` calls. The
:meth:`~NewtonActuatorAdapter.from_usd` classmethod parses
``NewtonActuator`` USD prims for PhysX and OVPhysX. Newton populates
``model.actuators`` itself.

DR gain updates bypass the adapter — the articulation writes straight
to controller arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import torch
import warp as wp
from newton.actuators import Actuator, Clamping, Delay

from .kernels import (
    build_implicit_dof_mask,
    build_per_dof_env_mask_kernel,
    scatter_gain_kernel,
    set_mask_kernel,
    zero_at_indices_kernel,
)

if TYPE_CHECKING:
    from isaaclab.actuators import ActuatorBase

# ---------------------------------------------------------------------------
# Abstract base — backend-independent logic
# ---------------------------------------------------------------------------


class NewtonActuatorAdapter:
    """Adapter that wraps a list of :class:`newton.actuators.Actuator`.

    Owns the actuator-state lifecycle, DOF-to-actuator bookkeeping,
    stepping, reset, and the pre-clamp computed-effort buffer the
    in-graph telemetry kernel reads on the post-actuator hook.
    """

    @dataclass(frozen=True)
    class ArticulationBinding:
        """Newton fast-path init state for one articulation.

        Returned by :meth:`bind_articulation`. Bundles the implicit-DOF mask
        and the per-articulation view of the adapter's computed-effort buffer.
        """

        implicit_dof_mask: wp.array
        """Per-DOF mask consumed by ``sync_torque_telemetry``; ``1`` on implicit-actuator DOFs, ``0`` otherwise."""

        implicit_dof_mask_owner: torch.Tensor
        """Torch tensor owning the memory :attr:`implicit_dof_mask` aliases; keep referenced for the mask's lifetime."""

        computed_effort_view: wp.array
        """This articulation's slice of the adapter's pre-clamp computed-effort buffer, ``(num_envs, num_joints)``."""

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

        # Collect the set of local DOFs covered by some actuator. Only the
        # env-0 slice of each actuator's flat ``indices`` array is needed —
        # later envs are repeats with a constant ``num_joints`` stride.
        managed: set[int] = set()
        for act in actuators:
            all_indices = act.indices.numpy()
            num_per_act = len(all_indices) // num_envs
            for global_dof in all_indices[:num_per_act]:
                local_dof = global_dof - dof_offset
                if 0 <= local_dof < num_joints:
                    managed.add(local_dof)

        if len(managed) == num_joints:
            self.joint_indices: torch.Tensor | slice = slice(None)
        else:
            self.joint_indices = torch.tensor(sorted(managed), dtype=torch.int32, device=device)

        self._states_a = [act.state() for act in actuators]
        self._states_b = [act.state() for act in actuators]

        # Pre-clamp computed effort buffer. Each Newton actuator scatter-adds
        # its raw controller output to ``sim_control.joint_computed_f`` when
        # ``control_computed_output_attr`` is set; we route that to this
        # buffer so the post-actuator telemetry kernel can report the actual
        # computed (pre-clamp) effort instead of mirroring ``joint_f``. The
        # binding onto ``sim_control`` happens in :meth:`finalize`.
        self._computed_effort = wp.zeros(
            num_envs * num_joints,
            dtype=wp.float32,
            device=device,
        )
        self.computed_effort_2d = self._computed_effort.reshape((num_envs, num_joints))
        for act in actuators:
            act.control_computed_output_attr = "joint_computed_f"

    def finalize(self, sim_control: Any) -> None:
        """Bind the pre-clamp computed-effort buffer onto ``sim_control``.

        Args:
            sim_control: The ``sim_control`` object that will be passed
                to :meth:`step` for this adapter's lifetime. Newton's
                ``Control`` on the Newton backend, an
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
        """
        sim_control.joint_computed_f = self._computed_effort

    def step(self, sim_state: Any, sim_control: Any, dt: float) -> None:
        """Zero actuated DOFs, step all actuators, and swap state buffers.

        Args:
            sim_state: Object with ``joint_q``, ``joint_qd``, etc.
                Newton ``State`` on the Newton backend,
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            sim_control: Object with ``joint_f``, ``joint_target_q``, etc.
                Newton ``Control`` on the Newton backend,
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            dt: Physics timestep [s].
        """
        # Zero before scatter-add (actuators accumulate into this buffer).
        self._computed_effort.zero_()
        for act in self.actuators:
            wp.launch(
                zero_at_indices_kernel,
                dim=act.indices.shape[0],
                inputs=[sim_control.joint_f, act.indices],
            )
        for act, sa, sb in zip(self.actuators, self._states_a, self._states_b):
            act.step(sim_state, sim_control, sa, sb, dt=dt)
        self._swap_state_buffers()

    def _swap_state_buffers(self) -> None:
        """Advance the actuator state ping-pong after an eager step or graph replay."""
        self._states_a, self._states_b = self._states_b, self._states_a

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset actuator states for the given environments.

        Args:
            env_ids: Environment indices to reset. ``None`` (or
                ``slice(None)``, which IsaacLab callers sometimes pass)
                resets all environments. Otherwise expects a torch tensor
                or sequence of int indices.

        Newton's :meth:`Actuator.State.reset` expects a per-DOF boolean
        mask of length ``num_actuators`` (= ``num_envs * dofs_per_actuator``),
        not a per-env mask — each entry gates the corresponding column of
        the actuator's state buffers (delay queue, controller integral,
        etc.). We therefore build a per-actuator per-DOF mask from the
        env mask before delegating to each state.
        """
        if env_ids is None or env_ids == slice(None):
            for sa, sb in zip(self._states_a, self._states_b):
                if sa is not None:
                    sa.reset(None)
                if sb is not None:
                    sb.reset(None)
            return

        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            idx = wp.from_torch(env_ids.to(device=self._device).contiguous().to(torch.int32), dtype=wp.int32)
        else:
            if len(env_ids) == 0:
                return
            idx = wp.array(list(env_ids), dtype=wp.int32, device=self._device)
        env_mask = wp.zeros(self._num_envs, dtype=wp.bool, device=self._device)
        wp.launch(set_mask_kernel, dim=idx.shape[0], inputs=[env_mask, idx], device=self._device)

        for act, sa, sb in zip(self.actuators, self._states_a, self._states_b):
            per_dof_mask = wp.zeros(act.indices.shape[0], dtype=wp.bool, device=self._device)
            wp.launch(
                build_per_dof_env_mask_kernel,
                dim=act.indices.shape[0],
                inputs=[act.indices, env_mask, self._dof_offset, self.num_joints, per_dof_mask],
                device=self._device,
            )
            if sa is not None:
                sa.reset(per_dof_mask)
            if sb is not None:
                sb.reset(per_dof_mask)

    def bind_articulation(
        self,
        *,
        lab_actuators: dict[str, ActuatorBase],
        dof_offset: int,
        num_joints: int,
    ) -> ArticulationBinding:
        """Assemble the Newton fast-path init state for one articulation.

        Builds the implicit-DOF mask and slices this adapter's
        computed-effort buffer to the articulation's columns.

        Args:
            lab_actuators: The articulation's Isaac Lab actuator groups in
                public joint order. Only :class:`~isaaclab.actuators.ImplicitActuator`
                groups contribute to :attr:`ArticulationBinding.implicit_dof_mask`.
            dof_offset: Offset of this articulation's DOFs in the adapter's
                env-major global index space (``0`` on PhysX, view-dependent
                on Newton).
            num_joints: Articulation-local joint count. Distinct from
                :attr:`num_joints`, which is the whole-model per-env DOF
                stride used to lay out the actuator index arrays.

        Returns:
            The bundled :class:`ArticulationBinding` for this articulation.
        """
        implicit_dof_mask, implicit_dof_mask_owner = build_implicit_dof_mask(lab_actuators, num_joints, self._device)
        computed_effort_view = self.computed_effort_2d[:, dof_offset : dof_offset + num_joints]
        return self.ArticulationBinding(
            implicit_dof_mask=implicit_dof_mask,
            implicit_dof_mask_owner=implicit_dof_mask_owner,
            computed_effort_view=computed_effort_view,
        )

    @property
    def is_all_graphable(self) -> bool:
        """``True`` when all actuators are CUDA-graph-safe."""
        return len(self.actuators) > 0 and all(a.is_graphable() for a in self.actuators)

    @property
    def is_stateful(self) -> bool:
        """``True`` when any actuator maintains delay or controller state."""
        return any(a.is_stateful() for a in self.actuators)

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
        """Build an adapter from ``NewtonActuator`` prims authored on *stage*.

        This is the host-adapter counterpart of Newton's
        ``ModelBuilder.add_usd``. It reads the same prims and constructs matching
        :class:`~newton.actuators.Actuator` objects. Structurally compatible
        joints are merged into one actuator with per-DOF parameter arrays and
        combined indices. Newton backends use ``model.actuators`` instead.

        On PhysX and OVPhysX, :paramref:`joint_names` is in this adapter's local
        public order and defines the local indices assigned to parsed actuator targets.

        Args:
            stage: USD stage containing ``NewtonActuator`` prims.
            joint_names: All articulation joint names in adapter-local public order.
            num_envs: Number of environments.
            num_joints: Number of joints per environment.
            device: Warp device string, for example ``"cuda:0"``.
            articulation_prim_path: Root prim path of environment zero's
                articulation. When set, only prims under this subtree are
                considered; otherwise the whole stage is scanned.

        Returns:
            Adapter whose actuator indices use :paramref:`joint_names` order.

        Raises:
            ValueError: If no authored actuator targets a name in
                :paramref:`joint_names`.
        """
        actuators = _create_actuators_from_usd(
            stage,
            joint_names,
            num_envs,
            num_joints,
            device,
            articulation_prim_path=articulation_prim_path,
        )
        return cls(actuators, num_envs, num_joints, dof_offset=0, device=device)


# ---------------------------------------------------------------------------
# Per-articulation controller-gain projections.
# ---------------------------------------------------------------------------


def _actuator_local_joint_ids(
    actuator: Actuator,
    dof_offset: int,
    num_joints: int,
    env_stride: int,
) -> set[int]:
    """Return actuator joints that belong to one articulation."""
    local_joint_ids = {int(global_dof) % env_stride - dof_offset for global_dof in actuator.indices.numpy()}
    return {joint_id for joint_id in local_joint_ids if 0 <= joint_id < num_joints}


def read_newton_actuator_gain(
    actuators: list[Actuator],
    attr: Literal["kp", "kd"],
    num_envs: int,
    num_joints: int,
    dof_offset: int,
    env_stride: int,
    device: str,
    joint_user_to_backend_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one live Newton controller gain into public joint order.

    Args:
        actuators: Newton actuators visible to the model.
        attr: Controller gain name.
        num_envs: Number of articulation environments.
        num_joints: Articulation-local joint count.
        dof_offset: Offset of this articulation's DOFs in the model buffer.
        env_stride: Model DOFs per environment.
        device: Torch and Warp device.
        joint_user_to_backend_indices: Optional public-to-backend joint mapping.

    Returns:
        The live controller gain and a public-order mask that identifies joints
        covered by a controller exposing :paramref:`attr`.

    Raises:
        ValueError: If the requested joint mapping is not a complete permutation.
    """
    user_to_backend: tuple[int, ...] | None = None
    if joint_user_to_backend_indices is not None:
        user_to_backend = tuple(int(index) for index in joint_user_to_backend_indices)
        if sorted(user_to_backend) != list(range(num_joints)):
            raise ValueError(
                "joint_user_to_backend_indices must contain each backend joint index exactly once; "
                f"expected a permutation of 0..{num_joints - 1}, got {user_to_backend}."
            )

    covered = torch.zeros(num_joints, dtype=torch.bool, device=device)
    wp_device = wp.get_device(device)
    flat_gains = wp.zeros(num_envs * num_joints, dtype=wp.float32, device=wp_device)
    for actuator in actuators:
        local_joint_ids = _actuator_local_joint_ids(actuator, dof_offset, num_joints, env_stride)
        if not local_joint_ids:
            continue
        controller = actuator.controller
        if not hasattr(controller, attr):
            continue
        covered[list(local_joint_ids)] = True
        wp.launch(
            scatter_gain_kernel,
            dim=actuator.indices.shape[0],
            inputs=[
                getattr(controller, attr),
                flat_gains,
                actuator.indices,
                dof_offset,
                num_envs,
                num_joints,
                env_stride,
            ],
            device=wp_device,
        )

    gains = wp.to_torch(flat_gains.reshape((num_envs, num_joints)))
    if user_to_backend is not None:
        backend_column_indices = torch.tensor(user_to_backend, dtype=torch.long, device=device)
        gains = gains.index_select(1, backend_column_indices)
        covered = covered.index_select(0, backend_column_indices)
    return gains, covered


# ---------------------------------------------------------------------------
# PhysX-only USD parsing
# ---------------------------------------------------------------------------

_ResolvedComponent: TypeAlias = tuple[type, dict[str, Any]]
_ResolvedActuatorSpec: TypeAlias = tuple[int, type, dict[str, Any], list[_ResolvedComponent]]


def _actuator_signature(
    controller_class: type,
    controller_arguments: dict[str, Any],
    component_arguments: list[_ResolvedComponent],
) -> tuple:
    """Build Newton's structural grouping key for a parsed actuator spec."""

    def make_hashable(value: Any) -> Any:
        if isinstance(value, list | tuple):
            return tuple(make_hashable(item) for item in value)
        return value

    def shared_key(component_class: type, resolved: dict[str, Any]) -> tuple:
        shared_names = getattr(component_class, "SHARED_PARAMS", set())
        return tuple(sorted((name, make_hashable(resolved[name])) for name in shared_names if name in resolved))

    clamping_key: list[tuple] = []
    has_delay = False
    for comp_cls, resolved in component_arguments:
        if issubclass(comp_cls, Delay):
            has_delay = True
        elif issubclass(comp_cls, Clamping):
            clamping_key.append((comp_cls, shared_key(comp_cls, resolved)))

    return (controller_class, has_delay, tuple(clamping_key), shared_key(controller_class, controller_arguments))


def _tile_per_dof_arguments(
    arguments: list[dict[str, Any]],
    num_envs: int,
    dtype: type,
    device: wp.Device,
) -> dict[str, wp.array]:
    """Pack per-joint scalar arguments in environment-major order."""
    if not arguments:
        return {}

    numpy_dtype = np.int32 if dtype == wp.int32 else np.float32
    return {
        name: wp.array(
            np.tile(np.asarray([per_joint[name] for per_joint in arguments], dtype=numpy_dtype), num_envs),
            dtype=dtype,
            device=device,
        )
        for name in arguments[0]
    }


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

    Joints with the same controller and clamping structure are merged into
    one :class:`Actuator`. Scalar parameters (``kp``, ``kd``,
    ``saturation_effort``, delay, etc.) are packed per DOF. Parameters marked
    as ``SHARED_PARAMS`` (e.g. ``model_path``, ``lookup_positions``) remain
    part of the grouping key and are passed through directly.
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
        raise ValueError(f"No NewtonActuator prims found targeting any of: {joint_names}")

    groups: dict[tuple, list[_ResolvedActuatorSpec]] = defaultdict(list)
    for local_idx, parsed in sorted(parsed_per_joint.items()):
        controller_arguments = parsed.controller_class.resolve_arguments(dict(parsed.controller_kwargs))
        component_arguments = [
            (comp_cls, comp_cls.resolve_arguments(comp_kwargs)) for comp_cls, comp_kwargs in parsed.component_specs
        ]
        sig = _actuator_signature(parsed.controller_class, controller_arguments, component_arguments)
        groups[sig].append((local_idx, parsed.controller_class, controller_arguments, component_arguments))

    actuators = []
    for grouped_specs in groups.values():
        local_indices = [spec[0] for spec in grouped_specs]
        controller_class = grouped_specs[0][1]
        resolved_controllers = [spec[2] for spec in grouped_specs]
        resolved_components = [spec[3] for spec in grouped_specs]

        flat_indices = np.array(
            [idx + e * num_total_joints for e in range(num_envs) for idx in local_indices],
            dtype=np.uint32,
        )
        indices = wp.array(flat_indices, device=wp_device)

        # Controller
        shared_ctrl = getattr(controller_class, "SHARED_PARAMS", set())
        ctrl_arguments = [
            {key: value for key, value in resolved.items() if key not in shared_ctrl}
            for resolved in resolved_controllers
        ]
        ctrl_shared = {key: value for key, value in resolved_controllers[0].items() if key in shared_ctrl}
        controller = controller_class(
            **_tile_per_dof_arguments(ctrl_arguments, num_envs, wp.float32, wp_device),
            **ctrl_shared,
        )

        # Components (delay + clampings)
        clamping_components = [
            [(comp_cls, resolved) for comp_cls, resolved in components if issubclass(comp_cls, Clamping)]
            for components in resolved_components
        ]
        delay_arguments = [
            resolved
            for components in resolved_components
            for comp_cls, resolved in components
            if issubclass(comp_cls, Delay)
        ]

        delay = None
        if delay_arguments:
            max_delay = max(int(arguments["delay_steps"]) for arguments in delay_arguments)
            if max_delay > 0:
                delay = Delay(
                    **_tile_per_dof_arguments(delay_arguments, num_envs, wp.int32, wp_device),
                    max_delay=max_delay,
                )

        clampings = []
        for component_index, (comp_cls, _) in enumerate(clamping_components[0]):
            resolved_clampings = [components[component_index][1] for components in clamping_components]
            shared_clamp = getattr(comp_cls, "SHARED_PARAMS", set())
            clamp_arguments = [
                {key: value for key, value in resolved.items() if key not in shared_clamp}
                for resolved in resolved_clampings
            ]
            clamp_shared = {key: value for key, value in resolved_clampings[0].items() if key in shared_clamp}
            clampings.append(
                comp_cls(
                    **_tile_per_dof_arguments(clamp_arguments, num_envs, wp.float32, wp_device),
                    **clamp_shared,
                )
            )

        actuator = Actuator(
            indices=indices,
            controller=controller,
            delay=delay,
            clamping=clampings if clampings else None,
            control_target_pos_attr="joint_target_pos",
            control_target_vel_attr="joint_target_vel",
        )
        actuators.append(actuator)

    return actuators
