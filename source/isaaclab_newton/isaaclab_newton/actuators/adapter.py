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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from newton.actuators import Actuator, Clamping, Delay

from .kernels import (
    build_implicit_dof_mask,
    fill_at_indices_kernel,
    gather_env_mask_kernel,
    scatter_flat_kernel,
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
        device: str,
        *,
        dof_count: int,
        dof_env_id: wp.array,
    ):
        """
        Args:
            actuators: Newton actuators over the model's flat DOF layout.
            num_envs: Number of environments (for reset masks).
            device: Warp device string.
            dof_count: Total flat DOF count of the model. All adapter buffers
                are flat over this index space — no per-env rectangle is
                assumed, so heterogeneous layouts work unchanged.
            dof_env_id: ``int32`` array of length :paramref:`dof_count` mapping
                each flat DOF to its environment index (``-1`` when the DOF
                belongs to no environment).
        """
        self.actuators = actuators
        self._num_envs = num_envs
        self._device = device
        self._dof_count = dof_count
        self._dof_env_id = dof_env_id

        # Flat snapshots over the global DOF space: initial gains and the
        # managed-DOF mask, filled by pure scatters through each actuator's
        # flat ``indices`` — layout-agnostic by construction.
        self._kp_flat = wp.zeros(dof_count, dtype=wp.float32, device=device)
        self._kd_flat = wp.zeros(dof_count, dtype=wp.float32, device=device)
        self._managed_flat = wp.zeros(dof_count, dtype=wp.bool, device=device)
        for act in actuators:
            ctrl = act.controller
            if hasattr(ctrl, "kp"):
                wp.launch(scatter_flat_kernel, dim=act.indices.shape[0], inputs=[ctrl.kp, act.indices, self._kp_flat])
            if hasattr(ctrl, "kd"):
                wp.launch(scatter_flat_kernel, dim=act.indices.shape[0], inputs=[ctrl.kd, act.indices, self._kd_flat])
            wp.launch(fill_at_indices_kernel, dim=act.indices.shape[0], inputs=[self._managed_flat, act.indices, True])

        # Every actuated DOF must have exactly one writer: overlapping
        # actuator index sets would make the scatter order load-bearing and
        # silently corrupt efforts. (~10 lines, checked once at build.)
        if actuators:
            all_indices = np.concatenate([act.indices.numpy() for act in actuators])
            counts = np.bincount(all_indices, minlength=dof_count)
            dup = np.nonzero(counts > 1)[0]
            if dup.size:
                raise ValueError(
                    f"NewtonActuatorAdapter: DOFs {dup[:8].tolist()} are claimed by more than one"
                    " actuator; every actuated DOF must have exactly one writer."
                )

        # Preallocated reset scratch (no allocation at reset-rate).
        self._reset_env_mask = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self._reset_dof_masks = [wp.zeros(act.indices.shape[0], dtype=wp.bool, device=device) for act in actuators]

        self._states_a = [act.state() for act in actuators]
        self._states_b = [act.state() for act in actuators]
        # Reset-event memo: a scene reset chain calls reset() once per
        # articulation with the same env set; the adapter state is
        # model-global, so repeats within one event are redundant.
        self._last_reset_key = None

        # Pre-clamp computed effort buffer, flat over the global DOF space.
        # Each Newton actuator scatter-adds its raw controller output here
        # (routed via ``control_computed_output_attr``) so the post-actuator
        # telemetry kernel can report the actual computed (pre-clamp) effort
        # instead of mirroring ``joint_f``. The binding onto ``sim_control``
        # happens in :meth:`finalize`; per-articulation strided views are
        # built by :meth:`bind_articulation`.
        self._computed_effort = wp.zeros(dof_count, dtype=wp.float32, device=device)
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
                fill_at_indices_kernel,
                dim=act.indices.shape[0],
                inputs=[sim_control.joint_f, act.indices, 0.0],
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
        from isaaclab.physics import PhysicsManager  # noqa: PLC0415

        if isinstance(env_ids, torch.Tensor):
            reset_key = (float(PhysicsManager._sim_time), tuple(env_ids.flatten().tolist()))
        elif env_ids is None or env_ids == slice(None):
            reset_key = (float(PhysicsManager._sim_time), None)
        else:
            reset_key = (float(PhysicsManager._sim_time), tuple(int(i) for i in env_ids))
        if reset_key == self._last_reset_key:
            return
        self._last_reset_key = reset_key

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
        env_mask = self._reset_env_mask
        env_mask.zero_()
        wp.launch(fill_at_indices_kernel, dim=idx.shape[0], inputs=[env_mask, idx, True], device=self._device)

        for act, sa, sb, per_dof_mask in zip(self.actuators, self._states_a, self._states_b, self._reset_dof_masks):
            wp.launch(
                gather_env_mask_kernel,
                dim=act.indices.shape[0],
                inputs=[act.indices, env_mask, self._dof_env_id, per_dof_mask],
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
        dof_index_map: torch.Tensor,
        joint_user_to_backend_indices: Sequence[int] | None = None,
    ) -> ArticulationBinding:
        """Assemble the Newton fast-path init state for one articulation.

        All per-articulation quantities are pure gathers of the adapter's flat
        global-DOF snapshots through :paramref:`dof_index_map` — no layout
        assumption beyond the map itself, so homogeneous and heterogeneous
        scenes take the same path.

        Args:
            lab_actuators: The articulation's Isaac Lab actuator groups in
                public joint order. Only :class:`~isaaclab.actuators.ImplicitActuator`
                groups contribute to :attr:`ArticulationBinding.implicit_dof_mask`.
            dof_index_map: Integer tensor of shape ``(num_instances, num_joints)``
                holding each (instance, backend-local joint)'s absolute flat DOF
                index in the adapter's global index space.
            joint_user_to_backend_indices: Complete permutation from public
                joint indices to backend-local joint indices. ``None`` preserves
                backend-local order.

        Returns:
            The bundled :class:`ArticulationBinding` for this articulation.

        Raises:
            ValueError: If :paramref:`joint_user_to_backend_indices` is not a
                complete permutation of the map's joint columns.
            NotImplementedError: If the articulation's instances are not
                uniformly strided in the flat DOF space (the computed-effort
                telemetry view requires a strided layout).
        """
        dof_map = dof_index_map.to(device=self._device, dtype=torch.long)
        num_instances, num_joints = dof_map.shape

        stiffness, damping, joint_indices = build_newton_actuator_defaults(
            kp_flat=wp.to_torch(self._kp_flat),
            kd_flat=wp.to_torch(self._kd_flat),
            managed_flat=wp.to_torch(self._managed_flat),
            dof_index_map=dof_map,
            joint_user_to_backend_indices=joint_user_to_backend_indices,
        )
        implicit_dof_mask, implicit_dof_mask_owner = build_implicit_dof_mask(lab_actuators, num_joints, self._device)

        # The telemetry kernel reads computed effort as (instance, backend joint):
        # expose a strided zero-copy view of the flat buffer. Joint columns must
        # be contiguous and instances uniformly spaced — both hold for Lab
        # articulation views (equal instance spacing is the Newton-view
        # invariant); anything else is not representable as a view.
        contiguous_cols = bool(torch.all(dof_map == dof_map[:, :1] + torch.arange(num_joints, device=dof_map.device)))
        row_starts = dof_map[:, 0]
        row_strides = row_starts[1:] - row_starts[:-1] if num_instances > 1 else None
        uniform_rows = row_strides is None or bool(torch.all(row_strides == row_strides[0]))
        if not (contiguous_cols and uniform_rows):
            raise NotImplementedError(
                "NewtonActuatorAdapter.bind_articulation: this articulation's DOFs are not"
                " uniformly strided in the flat DOF space; a computed-effort view cannot be"
                " built. Instance spacing must be uniform (the Newton view invariant)."
            )
        row_stride = int(row_strides[0]) if row_strides is not None else num_joints
        computed_effort_view = wp.from_torch(
            torch.as_strided(
                wp.to_torch(self._computed_effort),
                size=(num_instances, num_joints),
                stride=(row_stride, 1),
                storage_offset=int(row_starts[0]),
            )
        )
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
        # PhysX layouts are the trivial env-major rectangle; express it as the
        # explicit flat tables the adapter core requires.
        dof_env_id = wp.array(np.repeat(np.arange(num_envs, dtype=np.int32), num_joints), dtype=wp.int32, device=device)
        return cls(actuators, num_envs, device, dof_count=num_envs * num_joints, dof_env_id=dof_env_id)


# ---------------------------------------------------------------------------
# Per-articulation initial-gain snapshot — consumed by
# ``randomize_actuator_gains`` to seed ``default_joint_*`` baselines.
# ---------------------------------------------------------------------------


def build_newton_actuator_defaults(
    kp_flat: torch.Tensor,
    kd_flat: torch.Tensor,
    managed_flat: torch.Tensor,
    dof_index_map: torch.Tensor,
    joint_user_to_backend_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | slice]:
    """Snapshot initial Newton actuator gains for one articulation.

    Pure gathers of the adapter's flat global-DOF snapshots through the
    articulation's DOF index map. Without
    :paramref:`joint_user_to_backend_indices` the outputs stay in
    backend-local joint order; with it, gains and managed indices are
    converted to public joint order.

    Args:
        kp_flat: Flat per-DOF stiffness snapshot over the global DOF space.
        kd_flat: Flat per-DOF damping snapshot over the global DOF space.
        managed_flat: Flat boolean mask of DOFs claimed by a Newton actuator.
        dof_index_map: Integer tensor ``(num_instances, num_joints)`` of
            absolute flat DOF indices per (instance, backend-local joint).
        joint_user_to_backend_indices: Complete permutation from public joint
            indices to backend-local joint indices. ``None`` preserves
            backend-local order.

    Returns:
        Tuple containing the following values:

        * ``stiffness``: Initial gains [N/m or N·m/rad, depending on joint
          type], shape ``(num_instances, num_joints)``, dtype ``torch.float32``.
        * ``damping``: Initial gains [N·s/m or N·m·s/rad, depending on joint
          type], same shape and dtype.
        * ``joint_indices``: ``slice(None)`` when every joint is managed;
          otherwise a ``torch.int32`` tensor of managed columns, in the same
          (backend-local or public) order as the gain tensors.

    Raises:
        ValueError: If :paramref:`joint_user_to_backend_indices` is not a
            complete permutation of all joint columns.
    """
    num_instances, num_joints = dof_index_map.shape
    device = dof_index_map.device

    columns: torch.Tensor | None = None
    if joint_user_to_backend_indices is not None:
        user_to_backend = tuple(int(index) for index in joint_user_to_backend_indices)
        if sorted(user_to_backend) != list(range(num_joints)):
            raise ValueError(
                "joint_user_to_backend_indices must contain each backend joint index exactly once; "
                f"expected a permutation of 0..{num_joints - 1}, got {user_to_backend}."
            )
        columns = torch.tensor(user_to_backend, dtype=torch.long, device=device)

    stiffness = kp_flat[dof_index_map]
    damping = kd_flat[dof_index_map]
    managed = managed_flat[dof_index_map]
    if columns is not None:
        stiffness = stiffness[:, columns].contiguous()
        damping = damping[:, columns].contiguous()
        managed = managed[:, columns]

    managed_any = managed.any(dim=0)
    if not bool(torch.equal(managed_any, managed.all(dim=0))):
        warnings.warn(
            "Newton actuator coverage differs across instances of one articulation;"
            " treating a joint as managed when any instance manages it.",
            UserWarning,
            stacklevel=2,
        )
    if bool(managed_any.all()):
        joint_indices: torch.Tensor | slice = slice(None)
    else:
        joint_indices = torch.nonzero(managed_any, as_tuple=False).flatten().to(torch.int32)
    return stiffness, damping, joint_indices


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
