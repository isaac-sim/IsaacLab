# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-actuator adapter shared by Newton, PhysX, and OVPhysX.

Owns the actuator-state lifecycle, the supply batteries, the pre-clamp
computed-effort buffer, and the per-step ``step`` / ``reset`` / ``finalize``
calls. The
:meth:`~NewtonActuatorAdapter.from_usd` classmethod parses
``NewtonActuator`` USD prims for PhysX and OVPhysX. Newton populates
``model.actuators`` itself.

DR gain updates bypass the adapter — the articulation writes straight
to drive arrays.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch
import warp as wp
from newton import Model
from newton._src.utils.selection import FrequencyLayout
from newton.actuators import (
    Actuator,
    Battery,
    ClampingBase,
    InputProcessorBase,
    InputProcessorDelay,
    InputProcessorRandomDelay,
    parse_actuator_prim,
)
from newton.selection import ArticulationView

from ...utils import index_fill_
from .kernels import (
    build_implicit_dof_mask,
    build_per_dof_env_mask_kernel,
    set_mask_kernel,
    zero_at_indices_kernel,
)

if TYPE_CHECKING:
    from .. import ActuatorCollection

_SUPPLY_VOLTAGE = "supply_voltage"
"""Name of the ``sim_control`` array a :class:`~newton.actuators.Battery` writes and supply-reading drives read."""


@dataclass(frozen=True)
class _BatteryLoad:
    """The arrays :meth:`newton.actuators.Battery.refresh` reads and writes, in place of a ``Control``."""

    joint_f: wp.array
    """Effort drawn from the batteries [N·m], flat over the adapter's DOFs."""

    supply_voltage: wp.array
    """Supply voltage of each DOF's battery [V], flat over the adapter's DOFs. Written by the refresh."""


# ---------------------------------------------------------------------------
# Abstract base — backend-independent logic
# ---------------------------------------------------------------------------


class NewtonActuatorAdapter:
    """Adapter that wraps a list of :class:`newton.actuators.Actuator`.

    Owns the actuator-state lifecycle, DOF-to-actuator bookkeeping, the
    supply batteries, stepping, reset, and the pre-clamp computed-effort
    buffer the in-graph telemetry kernel reads on the post-actuator hook.
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
        # its raw drive output to ``sim_control.joint_computed_f`` when
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

        self._battery: Battery | None = None
        # First DOF of each battery, which locates its environment for masked resets.
        self._battery_dofs: wp.array | None = None
        self._battery_mask: wp.array | None = None
        self._reset_callbacks: list[Callable[[wp.array | None], None]] = []

    @property
    def battery(self) -> Battery | None:
        """Batteries refreshed before every actuator step, or None until :meth:`add_batteries` is called."""
        return self._battery

    def add_batteries(
        self,
        dof_indices: np.ndarray,
        nominal_voltage: wp.array,
        sag_gain: wp.array,
        min_voltage: float,
    ) -> int:
        """Add one battery per environment, shared by the given DOFs of that environment.

        Before the actuators, :meth:`step` computes each battery's voltage from the pre-clamp effort
        its DOFs drew on the previous step and writes it into ``sim_control.supply_voltage``. Drives
        that read the supply from there, such as :class:`~newton.actuators.DriveBAM` with
        ``supply_from_control=True``, then use it instead of their own supply voltage. :meth:`reset`
        restores the batteries of the reset environments. Add batteries before the first step: this
        reallocates the battery arrays.

        Args:
            dof_indices: DOF indices in the layout of the actuators' ``indices``, shape
                ``(num_envs, n)``. Row ``e`` lists the DOFs powered by environment ``e``'s battery.
            nominal_voltage: Unloaded voltage of each new battery [V], shape ``(num_envs,)``.
            sag_gain: Voltage drop per unit of total motor torque [V/(N·m)], shape ``(num_envs,)``.
            min_voltage: Lower bound of the sagged voltage [V].

        Returns:
            Index of environment 0's new battery; environment ``e``'s battery is at this index plus ``e``.

        Raises:
            ValueError: If the shapes do not match the environment count or a DOF already has a battery.
        """
        dof_indices = np.asarray(dof_indices, dtype=np.int64)
        nominal = nominal_voltage.numpy()
        gain = sag_gain.numpy()
        if dof_indices.ndim != 2 or dof_indices.shape[0] != self._num_envs:
            raise ValueError(f"dof_indices must have shape ({self._num_envs}, n), got {dof_indices.shape}")
        if nominal.shape != (self._num_envs,) or gain.shape != (self._num_envs,):
            raise ValueError(f"nominal_voltage and sag_gain must have shape ({self._num_envs},)")
        if self._battery is None:
            dof_battery = np.full(self._num_envs * self.num_joints, -1, dtype=np.int32)
            arrays = [np.empty(0, dtype=np.float32) for _ in range(3)]
            first_dofs = np.empty(0, dtype=np.uint32)
        else:
            battery = self._battery
            dof_battery = battery.dof_battery.numpy()
            arrays = [battery.nominal_voltage.numpy(), battery.sag_gain.numpy(), battery.min_voltage.numpy()]
            first_dofs = self._battery_dofs.numpy()
        if (dof_battery[dof_indices] >= 0).any():
            raise ValueError("Each DOF can be powered by one battery only.")
        offset = len(arrays[0])
        dof_battery[dof_indices] = offset + np.arange(self._num_envs, dtype=np.int32)[:, None]
        minimum = np.full(self._num_envs, min_voltage, dtype=np.float32)
        nominal, gain, minimum = (
            np.concatenate([old, added.astype(np.float32)]) for old, added in zip(arrays, (nominal, gain, minimum))
        )
        self._battery = Battery(
            dof_battery=wp.array(dof_battery, dtype=wp.int32, device=self._device),
            nominal_voltage=wp.array(nominal, dtype=wp.float32, device=self._device),
            sag_gain=wp.array(gain, dtype=wp.float32, device=self._device),
            min_voltage=wp.array(minimum, dtype=wp.float32, device=self._device),
        )
        first_dofs = np.concatenate([first_dofs, dof_indices[:, 0].astype(np.uint32)])
        self._battery_dofs = wp.array(first_dofs, dtype=wp.uint32, device=self._device)
        self._battery_mask = wp.zeros(len(first_dofs), dtype=wp.bool, device=self._device)
        return offset

    def register_reset_callback(self, callback: Callable[[wp.array | None], None]) -> None:
        """Append a hook that :meth:`reset` calls after resetting the actuator states and batteries.

        Hooks reset per-environment state kept outside the Newton actuators. They are called on
        every reset of at least one environment, in registration order.

        Args:
            callback: Called with the per-environment boolean reset mask, shape ``(num_envs,)``, or
                None when every environment is reset.
        """
        self._reset_callbacks.append(callback)

    def finalize(self, sim_control: Any) -> None:
        """Bind the pre-clamp computed-effort buffer onto ``sim_control``, and the supply voltage if missing.

        Newton's model builder registers ``supply_voltage`` on its ``Control`` when a drive reads the
        supply from there. Other ``sim_control`` objects get a zeroed array from the adapter.

        Args:
            sim_control: The ``sim_control`` object that will be passed
                to :meth:`step` for this adapter's lifetime. Newton's
                ``Control`` on the Newton backend, an
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
        """
        sim_control.joint_computed_f = self._computed_effort
        reads_supply = any(_SUPPLY_VOLTAGE in act.drive.custom_inputs for act in self.actuators)
        if reads_supply and getattr(sim_control, _SUPPLY_VOLTAGE, None) is None:
            setattr(sim_control, _SUPPLY_VOLTAGE, wp.zeros_like(self._computed_effort))

    def step(self, sim_state: Any, sim_control: Any, dt: float, *, swap_state: bool = True) -> None:
        """Refresh the batteries, zero actuated DOFs, step all actuators, and advance state buffers.

        Args:
            sim_state: Object with ``joint_q``, ``joint_qd``, etc.
                Newton ``State`` on the Newton backend,
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            sim_control: Object with ``joint_f``, ``joint_target_q``, etc.
                Newton ``Control`` on the Newton backend,
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend. Needs ``supply_voltage`` once batteries are added.
            dt: Physics timestep [s].
            swap_state: Swap state buffers after stepping. Set to ``False`` on the last step
                of an odd-length graph to copy the output back into the input buffers instead,
                keeping their addresses stable across replays.
        """
        # The supply voltage of this step follows from the motor torque drawn on the previous one, before the
        # effort clamp as in the BAM reference. sim_control.joint_f does not keep it: the articulations write their
        # effort targets there before each simulation step.
        if self._battery is not None:
            self._battery.refresh(_BatteryLoad(self._computed_effort, getattr(sim_control, _SUPPLY_VOLTAGE, None)))
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
        if swap_state:
            self._swap_state_buffers()
        else:
            for sa, sb in zip(self._states_a, self._states_b):
                if sa is not None:
                    sa.assign(sb)

    def _swap_state_buffers(self) -> None:
        """Advance the actuator state ping-pong after an eager step or graph replay."""
        self._states_a, self._states_b = self._states_b, self._states_a

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Reset actuator states, batteries, and registered reset hooks for the given environments.

        Args:
            env_ids: Environment indices to reset. ``None`` (or
                ``slice(None)``, which IsaacLab callers sometimes pass)
                resets all environments. Otherwise expects a torch tensor
                or sequence of int indices.

        Newton's :meth:`Actuator.State.reset` expects a per-DOF boolean
        mask of length ``num_actuators`` (= ``num_envs * dofs_per_actuator``),
        not a per-env mask — each entry gates the corresponding column of
        the actuator's state buffers (delay queue, drive integral,
        etc.). We therefore build a per-actuator per-DOF mask from the
        env mask before delegating to each state.
        """
        if env_ids is None or env_ids == slice(None):
            env_mask = None
            for sa, sb in zip(self._states_a, self._states_b):
                if sa is not None:
                    sa.reset(None)
                if sb is not None:
                    sb.reset(None)
        else:
            num_envs = len(range(self._num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
            if num_envs == 0:
                return
            env_mask = wp.zeros(self._num_envs, dtype=wp.bool, device=self._device)
            if isinstance(env_ids, slice):
                wp.to_torch(env_mask)[env_ids] = True
            else:
                # Torch indexed scalar assignment uploads True and synchronizes on CUDA.
                indices = (
                    wp.from_torch(env_ids)
                    if isinstance(env_ids, torch.Tensor)
                    else wp.array(env_ids, dtype=wp.int32, device=self._device)
                )
                wp.launch(set_mask_kernel, dim=num_envs, inputs=[env_mask, indices], device=self._device)

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

        if self._battery is not None:
            battery_mask = None
            if env_mask is not None:
                battery_mask = self._battery_mask
                wp.launch(
                    build_per_dof_env_mask_kernel,
                    dim=len(battery_mask),
                    inputs=[self._battery_dofs, env_mask, self._dof_offset, self.num_joints, battery_mask],
                    device=self._device,
                )
            self._battery.reset(battery_mask)
        for callback in self._reset_callbacks:
            callback(env_mask)

    def bind_articulation(
        self,
        *,
        implicit_joint_indices: Sequence[slice | torch.Tensor | None],
        dof_offset: int,
        num_joints: int,
    ) -> ArticulationBinding:
        """Assemble the Newton fast-path init state for one articulation.

        Builds the implicit-DOF mask and slices this adapter's
        computed-effort buffer to the articulation's columns.

        Args:
            implicit_joint_indices: Joint selectors of the articulation's implicit
                actuator groups in public joint order; they define
                :attr:`ArticulationBinding.implicit_dof_mask`.
            dof_offset: Offset of this articulation's DOFs in the adapter's
                env-major global index space (``0`` on PhysX, view-dependent
                on Newton).
            num_joints: Articulation-local joint count. Distinct from
                :attr:`num_joints`, which is the whole-model per-env DOF
                stride used to lay out the actuator index arrays.

        Returns:
            The bundled :class:`ArticulationBinding` for this articulation.
        """
        implicit_dof_mask, implicit_dof_mask_owner = build_implicit_dof_mask(
            implicit_joint_indices, num_joints, self._device
        )
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
        """``True`` when any actuator maintains delay or drive state."""
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
# Component-addressed parameter access via Newton's selection API.
# ---------------------------------------------------------------------------


def read_group_parameter(collection: ActuatorCollection, name: str, component: str, attr: str) -> torch.Tensor:
    """Read one live Newton actuator parameter for a native group.

    Group-scoped, user-ordered reads of the drive-owned storage. For raw
    component access, use the group's Newton actuator object (the collection
    mapping entry) directly.

    Args:
        collection: The articulation's actuator collection.
        name: Actuator group name.
        component: Component kind: ``"drive"``, ``"delay"``, or ``"clamping"``.
            ``"controller"`` is a deprecated alias for ``"drive"``.
        attr: Parameter name on that component (e.g. ``"kp"``, ``"max_effort"``).

    Returns:
        Live values in the group's joint order, shape
        ``(num_instances, group_num_joints)``, in the parameter's dtype.
        Units follow the addressed parameter.

    Raises:
        ValueError: If the group is not executed by Newton actuators, the
            component name is unknown, or no actuator exposes the parameter.
    """
    owners = _group_parameter_owners(collection, name, component, attr)
    view = collection._newton_selection.view
    values: torch.Tensor | None = None
    for actuator, owner in owners:
        # Non-driven DOFs read as zeros, and groups are disjoint, so overlaying is a sum.
        projected = wp.to_torch(view.get_actuator_parameter(actuator, owner, attr))
        values = projected if values is None else values + projected
    return values[:, collection._newton_group_columns(name)]


def write_group_parameter(
    collection: ActuatorCollection,
    name: str,
    component: str,
    attr: str,
    values: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
    joint_ids: torch.Tensor | None = None,
) -> None:
    """Write one Newton actuator parameter for a native group.

    Group-scoped, user-ordered writes that reach the drive-owned storage
    through Newton's selection API. For raw component access, use the group's
    Newton actuator object (the collection mapping entry) directly.

    Args:
        collection: The articulation's actuator collection.
        name: Actuator group name.
        component: Component kind: ``"drive"``, ``"delay"``, or ``"clamping"``.
            ``"controller"`` is a deprecated alias for ``"drive"``.
        attr: Parameter name on that component (e.g. ``"kp"``, ``"max_effort"``).
        values: New values, shape ``(len(env_ids), len(joint_ids))``. Units
            follow the addressed parameter.
        env_ids: Environment indices to update. Defaults to all environments.
        joint_ids: Group-local joint indices to update. Defaults to all of
            the group's joints.

    Raises:
        ValueError: Same conditions as :func:`read_group_parameter`.
    """
    owners = _group_parameter_owners(collection, name, component, attr)
    view = collection._newton_selection.view
    device = collection.device
    columns = collection._newton_group_columns(name)
    if joint_ids is not None:
        columns = columns[joint_ids.to(device, dtype=torch.long)]
    mask = None
    env_rows: torch.Tensor | slice | None = None
    if env_ids is not None and env_ids != slice(None):
        env_rows = env_ids if isinstance(env_ids, slice) else env_ids[:, None]
        mask_torch = torch.zeros(collection.num_instances, dtype=torch.bool, device=device)
        index_fill_(mask_torch, env_ids, True)
        mask = wp.from_torch(mask_torch, dtype=wp.bool)
    values = values.to(device)
    for actuator, owner in owners:
        current = view.get_actuator_parameter(actuator, owner, attr)
        current_torch = wp.to_torch(current)
        if env_rows is None:
            current_torch[:, columns] = values.to(dtype=current_torch.dtype)
        else:
            current_torch[env_rows, columns] = values.to(dtype=current_torch.dtype)
        view.set_actuator_parameter(actuator=actuator, component=owner, name=attr, values=current, mask=mask)


def _group_parameter_owners(
    collection: ActuatorCollection, name: str, component: str, attr: str
) -> list[tuple[Actuator, Any]]:
    """Resolve the component instances exposing ``attr`` for one native group."""
    if component == "controller":
        # stacklevel=3 points at the caller of read_group_parameter / write_group_parameter.
        warnings.warn(
            "The 'controller' actuator component is deprecated. Use 'drive' instead; "
            "'controller' will be removed in 3.2.",
            DeprecationWarning,
            stacklevel=3,
        )
        component = "drive"
    if name not in collection._groups:
        raise KeyError(name)
    if collection._newton_selection is None or name not in collection._native_group_names:
        raise ValueError(f"Actuator group '{name}' is not executed by Newton actuators.")
    group_actuators = collection._groups[name]
    if not isinstance(group_actuators, tuple):
        group_actuators = (group_actuators,)
    owners = [
        (actuator, owner)
        for actuator in group_actuators
        if (owner := resolve_actuator_component(actuator, component, attr)) is not None
    ]
    if not owners:
        raise ValueError(f"No Newton actuator exposes parameter ('{component}', '{attr}').")
    return owners


def resolve_actuator_component(actuator: Actuator, component: str, attr: str) -> Any | None:
    """Return the component instance that exposes ``attr`` on the addressed component kind.

    ``component`` selects the actuator's ``"drive"``, ``"delay"``, or
    ``"clamping"`` entry; the returned object is what Newton's
    :meth:`~newton.selection.ArticulationView.get_actuator_parameter` and
    :meth:`~newton.selection.ArticulationView.set_actuator_parameter` take as
    their ``component`` argument. Returns ``None`` when the component is absent
    on this actuator or does not expose ``attr``. Raises ``ValueError`` on
    unknown component names or ambiguous clamping matches.
    """
    if component == "drive":
        owner = actuator.drive
    elif component == "delay":
        owner = next((p for p in actuator.input_processors if isinstance(p, InputProcessorDelay)), None)
    elif component == "clamping":
        matches = [entry for entry in (getattr(actuator, "clamping", None) or []) if hasattr(entry, attr)]
        if len(matches) > 1:
            names = ", ".join(type(entry).__name__ for entry in matches)
            raise ValueError(f"Ambiguous clamping parameter '{attr}': exposed by {names}.")
        owner = matches[0] if matches else None
    else:
        raise ValueError(f"Unknown actuator component '{component}'. Expected 'drive', 'delay', or 'clamping'.")
    if owner is None or not hasattr(owner, attr):
        return None
    return owner


class LightArticulationView:
    """Newton's actuator-parameter selection over bare actuators, without a Model.

    The PhysX-family backends build Newton actuators from USD without a Newton
    :class:`~newton.Model`, so they cannot construct a real
    :class:`~newton.selection.ArticulationView`. This stand-in provides the placement
    attributes and DOF mapping cache for the PhysX flat layout (one articulation per world,
    identity joint order, per-world DOF stride equal to the joint count) and
    borrows the real implementations unchanged.
    """

    def __init__(self, num_envs: int, num_joints: int, device: str):
        self.world_count = num_envs
        self.count_per_world = 1
        self.device = device
        self.full_mask = wp.ones(num_envs, dtype=wp.bool, device=device)
        self._actuator_dof_mapping_cache = {}
        self.frequency_layouts = {
            Model.AttributeFrequency.JOINT_DOF: FrequencyLayout(
                offset=0,
                stride_between_worlds=num_joints,
                stride_within_worlds=num_joints,
                value_count=num_joints,
                indices=list(range(num_joints)),
                device=device,
            )
        }

    get_actuator_parameter = ArticulationView.get_actuator_parameter
    set_actuator_parameter = ArticulationView.set_actuator_parameter
    _get_actuator_dof_mapping = ArticulationView._get_actuator_dof_mapping
    _create_actuator_dof_mapping = ArticulationView._create_actuator_dof_mapping
    _resolve_world_mask = ArticulationView._resolve_world_mask


@dataclass(frozen=True)
class NewtonActuatorSelection:
    """Execution-setup handoff for Newton actuator parameter access.

    Pure data returned by
    :meth:`~isaaclab.actuators.ActuatorControl.finalize_native_actuators` and
    consumed by the group-scoped parameter access functions
    (:func:`read_group_parameter` / :func:`write_group_parameter`) and by the
    collection when it maps native groups to their Newton actuator objects.
    """

    view: Any
    """Newton :class:`~newton.selection.ArticulationView` or :class:`LightArticulationView` over the articulation."""

    actuators: list[Actuator]
    """Newton actuators visible to the view."""

    joint_user_to_backend_indices: tuple[int, ...] | None = None
    """Optional public-to-backend joint permutation for the view's DOF columns."""


# ---------------------------------------------------------------------------
# PhysX-only USD parsing
# ---------------------------------------------------------------------------

_ResolvedComponent: TypeAlias = tuple[type, dict[str, Any]]
_ResolvedActuatorSpec: TypeAlias = tuple[int, type, dict[str, Any], list[_ResolvedComponent]]


def _actuator_signature(
    drive_class: type,
    drive_arguments: dict[str, Any],
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

    processor_key: list[tuple] = []
    clamping_key: list[tuple] = []
    for comp_cls, resolved in component_arguments:
        if issubclass(comp_cls, InputProcessorBase):
            processor_key.append((comp_cls, shared_key(comp_cls, resolved)))
        elif issubclass(comp_cls, ClampingBase):
            clamping_key.append((comp_cls, shared_key(comp_cls, resolved)))

    return (drive_class, tuple(processor_key), tuple(clamping_key), shared_key(drive_class, drive_arguments))


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

    Joints with the same drive and clamping structure are merged into
    one :class:`Actuator`. Scalar parameters (``kp``, ``kd``,
    ``saturation_effort``, delay, etc.) are packed per DOF. Parameters marked
    as ``SHARED_PARAMS`` (e.g. ``model_path``, ``lookup_positions``) remain
    part of the grouping key and are passed through directly.
    """
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
        drive_arguments = parsed.drive_class.resolve_arguments(dict(parsed.drive_kwargs))
        component_arguments = [
            (comp_cls, comp_cls.resolve_arguments(comp_kwargs)) for comp_cls, comp_kwargs in parsed.component_specs
        ]
        sig = _actuator_signature(parsed.drive_class, drive_arguments, component_arguments)
        groups[sig].append((local_idx, parsed.drive_class, drive_arguments, component_arguments))

    actuators = []
    for grouped_specs in groups.values():
        local_indices = [spec[0] for spec in grouped_specs]
        drive_class = grouped_specs[0][1]
        resolved_drives = [spec[2] for spec in grouped_specs]
        resolved_components = [spec[3] for spec in grouped_specs]

        flat_indices = np.array(
            [idx + e * num_total_joints for e in range(num_envs) for idx in local_indices],
            dtype=np.uint32,
        )
        indices = wp.array(flat_indices, device=wp_device)

        # Drive
        shared_drive = getattr(drive_class, "SHARED_PARAMS", set())
        drive_arguments = [
            {key: value for key, value in resolved.items() if key not in shared_drive} for resolved in resolved_drives
        ]
        drive_shared = {key: value for key, value in resolved_drives[0].items() if key in shared_drive}
        drive = drive_class(
            **_tile_per_dof_arguments(drive_arguments, num_envs, wp.float32, wp_device),
            **drive_shared,
        )

        # Components (input processors + clampings)
        clamping_components = [
            [(comp_cls, resolved) for comp_cls, resolved in components if issubclass(comp_cls, ClampingBase)]
            for components in resolved_components
        ]
        processor_components = [
            [(comp_cls, resolved) for comp_cls, resolved in components if issubclass(comp_cls, InputProcessorBase)]
            for components in resolved_components
        ]

        input_processors = []
        for component_index, (comp_cls, _) in enumerate(processor_components[0]):
            delay_arguments = [components[component_index][1] for components in processor_components]
            if comp_cls is InputProcessorRandomDelay:
                # One lag stream per environment, as Newton's builder gives each world its own.
                lag_ranges = [
                    {name: args[name] for name in ("max_delay", "min_delay", "update_period")}
                    for args in delay_arguments
                ]
                holds = [{"hold_probability": args["hold_probability"]} for args in delay_arguments]
                world = np.repeat(np.arange(num_envs, dtype=np.int32), len(delay_arguments))
                input_processors.append(
                    InputProcessorRandomDelay(
                        **_tile_per_dof_arguments(lag_ranges, num_envs, wp.int32, wp_device),
                        **_tile_per_dof_arguments(holds, num_envs, wp.float32, wp_device),
                        world=wp.array(world, dtype=wp.int32, device=wp_device),
                        seed=delay_arguments[0]["seed"],
                    )
                )
                continue
            if comp_cls is not InputProcessorDelay:
                raise ValueError(
                    f"Input processor {comp_cls.__name__} is not supported without a Newton model; only"
                    " InputProcessorDelay and InputProcessorRandomDelay are."
                )
            max_delay = max(int(arguments["delay_steps"]) for arguments in delay_arguments)
            if max_delay > 0:
                input_processors.append(
                    InputProcessorDelay(
                        **_tile_per_dof_arguments(delay_arguments, num_envs, wp.int32, wp_device),
                        max_delay=max_delay,
                    )
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
            drive=drive,
            input_processors=input_processors,
            clamping=clampings if clampings else None,
            control_target_pos_attr="joint_target_pos",
            control_target_vel_attr="joint_target_vel",
        )
        actuators.append(actuator)

    return actuators
