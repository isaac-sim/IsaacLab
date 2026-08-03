# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-actuator adapter shared by the Newton and PhysX backends.

Owns the actuator-state lifecycle, the pre-clamp computed-effort buffer,
and the per-step ``step`` / ``reset`` / ``finalize`` calls. The
:meth:`~NewtonActuatorAdapter.from_usd` classmethod parses
``NewtonActuator`` USD prims on the PhysX backend (Newton populates
``model.actuators`` itself).

DR gain updates bypass the adapter — the articulation writes straight
to controller arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

        Returned by :meth:`bind_articulation`. Bundles the pieces the
        articulation formerly assembled from separate free-function calls:
        the initial gain snapshot, the implicit-DOF mask, and the
        per-articulation view of the adapter's computed-effort buffer.
        """

        stiffness: torch.Tensor
        """Initial stiffness gains [N/m or N·m/rad, depending on joint type], shape ``(num_envs, num_joints)``."""

        damping: torch.Tensor
        """Initial damping gains [N·s/m or N·m·s/rad, depending on joint type], shape ``(num_envs, num_joints)``."""

        joint_indices: torch.Tensor | slice
        """Managed columns; ``slice(None)`` when every joint is managed, else a ``torch.int32`` index tensor."""

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
                :class:`~isaaclab_newton.actuators.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
        """
        sim_control.joint_computed_f = self._computed_effort

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
        joint_user_to_backend_indices: Sequence[int] | None = None,
    ) -> ArticulationBinding:
        """Assemble the Newton fast-path init state for one articulation.

        Consolidates the pieces the articulation formerly built with
        separate :func:`build_newton_actuator_defaults` and
        :func:`build_implicit_dof_mask` calls plus a manual
        computed-effort slice: it snapshots the initial gains, builds the
        implicit-DOF mask, and slices this adapter's computed-effort buffer
        to the articulation's columns. Whole-model quantities
        (:attr:`actuators`, :attr:`num_joints` as the env stride,
        ``num_envs``, ``device``) come from the adapter; only the
        articulation-local placement varies per call.

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
            joint_user_to_backend_indices: Complete permutation from public
                joint indices to adapter-local joint indices. ``None``
                preserves adapter-local order (the PhysX case, whose adapter
                is already built from public joint names).

        Returns:
            The bundled :class:`ArticulationBinding` for this articulation.
        """
        stiffness, damping, joint_indices = build_newton_actuator_defaults(
            actuators=self.actuators,
            num_envs=self._num_envs,
            num_joints=num_joints,
            dof_offset=dof_offset,
            env_stride=self.num_joints,
            device=self._device,
            joint_user_to_backend_indices=joint_user_to_backend_indices,
        )
        implicit_dof_mask, implicit_dof_mask_owner = build_implicit_dof_mask(lab_actuators, num_joints, self._device)
        computed_effort_view = self.computed_effort_2d[:, dof_offset : dof_offset + num_joints]
        return self.ArticulationBinding(
            stiffness=stiffness,
            damping=damping,
            joint_indices=joint_indices,
            implicit_dof_mask=implicit_dof_mask,
            implicit_dof_mask_owner=implicit_dof_mask_owner,
            computed_effort_view=computed_effort_view,
        )

    @property
    def is_all_graphable(self) -> bool:
        """``True`` when all actuators are CUDA-graph-safe."""
        return len(self.actuators) > 0 and all(a.is_graphable() for a in self.actuators)

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

        This is the PhysX-side counterpart of Newton's
        ``ModelBuilder.add_usd``. It reads the same prims and constructs matching
        :class:`~newton.actuators.Actuator` objects. Joints with the same
        controller, gains, clamping, and delay are merged into one actuator with
        combined indices. Newton backends use ``model.actuators`` instead.

        On PhysX, :paramref:`joint_names` is in this adapter's local public order
        and defines the local indices assigned to parsed actuator targets.

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
# Per-articulation initial-gain snapshot — consumed by
# ``randomize_actuator_gains`` to seed ``default_joint_*`` baselines.
# ---------------------------------------------------------------------------


def build_newton_actuator_defaults(
    actuators: list[Actuator],
    num_envs: int,
    num_joints: int,
    dof_offset: int,
    env_stride: int,
    device: str,
    joint_user_to_backend_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | slice]:
    """Snapshot initial Newton actuator gains for one articulation.

    Actuators are filtered to those whose environment-zero DOF lies in
    ``[dof_offset, dof_offset + num_joints)``. Their gains are scattered in the
    actuator adapter's local joint order. Without :paramref:`joint_user_to_backend_indices`,
    the output preserves that local order. PhysX builds its per-articulation adapter from
    public joint names, so its adapter-local order is public order. Newton's global adapter
    uses backend-local order; the optional map converts its gains and managed indices to
    public order.

    Args:
        actuators: Newton actuators visible to this articulation.
        num_envs: Number of environments.
        num_joints: Articulation-local joint count.
        dof_offset: Offset of this articulation's DOFs in the env-major
            global index space (``0`` on PhysX, view-dependent on Newton).
        env_stride: Whole-model per-env DOF count — the stride used to build
            each actuator's env-major ``indices``. Equals ``num_joints`` on
            PhysX, but exceeds it by the free-root DOFs on a floating-base
            Newton articulation, so it must be passed explicitly rather than
            assumed equal to ``num_joints``. The owning adapter's
            :attr:`NewtonActuatorAdapter.num_joints` is exactly this value.
        device: Warp device string (e.g. ``"cuda:0"``).
        joint_user_to_backend_indices: Complete permutation from public joint
            indices to adapter-local joint indices. For Newton's global adapter,
            adapter-local order is backend order. ``None`` preserves adapter-local order.

    Returns:
        Tuple containing the following values:

        * ``stiffness``: Initial gains [N/m or N·m/rad, depending on joint
          type], shape ``(num_envs, num_joints)``, dtype ``torch.float32``, on
          :paramref:`device`.
        * ``damping``: Initial gains [N·s/m or N·m·s/rad, depending on joint
          type], shape ``(num_envs, num_joints)``, dtype ``torch.float32``, on
          :paramref:`device`.
        * ``joint_indices``: ``slice(None)`` when every joint is managed;
          otherwise, a ``torch.int32`` tensor on :paramref:`device` containing
          managed columns in the same adapter-local or public order as the gain tensors.

    Raises:
        ValueError: If :paramref:`joint_user_to_backend_indices` is not a
            complete permutation of all adapter-local joint indices.
    """
    user_to_backend: tuple[int, ...] | None = None
    if joint_user_to_backend_indices is not None:
        user_to_backend = tuple(int(index) for index in joint_user_to_backend_indices)
        if sorted(user_to_backend) != list(range(num_joints)):
            raise ValueError(
                "joint_user_to_backend_indices must contain each backend joint index exactly once; "
                f"expected a permutation of 0..{num_joints - 1}, got {user_to_backend}."
            )

    arti_actuators = [act for act in actuators if dof_offset <= int(act.indices.numpy()[0]) < dof_offset + num_joints]

    managed_local: set[int] = set()
    for act in arti_actuators:
        per_act = act.indices.shape[0] // num_envs
        for global_dof in act.indices.numpy()[:per_act]:
            local = int(global_dof) - dof_offset
            if 0 <= local < num_joints:
                managed_local.add(local)
    joint_indices: torch.Tensor | slice
    if len(managed_local) == num_joints:
        joint_indices = slice(None)
    else:
        joint_indices = torch.tensor(sorted(managed_local), dtype=torch.int32, device=device)

    wp_device = wp.get_device(device)
    flat_stiffness = wp.zeros(num_envs * num_joints, dtype=wp.float32, device=wp_device)
    flat_damping = wp.zeros(num_envs * num_joints, dtype=wp.float32, device=wp_device)
    for act in arti_actuators:
        ctrl = act.controller
        if hasattr(ctrl, "kp"):
            wp.launch(
                scatter_gain_kernel,
                dim=act.indices.shape[0],
                inputs=[ctrl.kp, flat_stiffness, act.indices, dof_offset, num_joints, env_stride],
                device=wp_device,
            )
        if hasattr(ctrl, "kd"):
            wp.launch(
                scatter_gain_kernel,
                dim=act.indices.shape[0],
                inputs=[ctrl.kd, flat_damping, act.indices, dof_offset, num_joints, env_stride],
                device=wp_device,
            )
    stiffness = wp.to_torch(flat_stiffness.reshape((num_envs, num_joints)))
    damping = wp.to_torch(flat_damping.reshape((num_envs, num_joints)))
    if user_to_backend is not None:
        # ``index_select(1, backend_column_indices)`` gathers backend-order columns into user-order
        # positions: for each user position ``u`` it holds the backend column ``user_to_backend[u]``.
        backend_column_indices = torch.tensor(user_to_backend, dtype=torch.long, device=device)
        stiffness = stiffness.index_select(1, backend_column_indices)
        damping = damping.index_select(1, backend_column_indices)
        if not isinstance(joint_indices, slice):
            backend_to_user = [0] * num_joints
            for user_index, backend_index in enumerate(user_to_backend):
                backend_to_user[backend_index] = user_index
            joint_indices = torch.tensor(
                sorted(backend_to_user[index] for index in managed_local),
                dtype=torch.int32,
                device=device,
            )
    return stiffness, damping, joint_indices


# ---------------------------------------------------------------------------
# PhysX-only USD parsing
# ---------------------------------------------------------------------------


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
        raise ValueError(f"No NewtonActuator prims found targeting any of: {joint_names}")

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
                            num_dofs_in_group,
                            float(v),
                            dtype=wp.float32,
                            device=wp_device,
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
