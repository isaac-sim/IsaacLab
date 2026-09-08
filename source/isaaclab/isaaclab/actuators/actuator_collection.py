# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime actuator collection for articulations."""

from __future__ import annotations

import copy
import itertools
import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence

import torch
import warp as wp
from prettytable import PrettyTable

from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache

from . import actuator_kernels
from ._compat import _resolve_limit_aliases
from .actuator_base import ActuatorBase, resolve_joint_parameter
from .actuator_base_cfg import ActuatorBaseCfg, _is_implicit_actuator_cfg
from .actuator_control import ActuatorControl
from .actuator_pd import IdealPDActuator, ImplicitActuator

logger = logging.getLogger(__name__)


class ActuatorCollection(Mapping[str, "ActuatorBase | object"]):
    """Read-only runtime collection of actuator groups for one articulation.

    Mapping entries return whoever owns the group. Isaac Lab-executed groups map to
    their :class:`~isaaclab.actuators.ActuatorBase` model instances. Newton-executed
    groups map to the Newton :class:`~newton.actuators.Actuator` objects that drive
    their joints, so users read and modify the owning controller directly. Newton
    merges structurally identical joints into one actuator, so several groups can
    map to the same object (or to a tuple when a group spans several); the
    collection keeps each group's joint indices, which
    :func:`~isaaclab.actuators.newton.read_group_parameter` and
    :func:`~isaaclab.actuators.newton.write_group_parameter` use for
    group-scoped, user-ordered access.

    Configure membership through :attr:`isaaclab.assets.ArticulationCfg.actuators`
    before construction; assigning or deleting mapping entries raises
    :class:`TypeError`. Each joint can belong to at most one group; overlapping
    joint selections raise :class:`ValueError` during construction.

    Plain :class:`~isaaclab.actuators.ImplicitActuator` groups are not executed one
    group at a time: a single internal executor computes all of their joints in one
    fused kernel launch. All other Lab-executed groups, including subclasses of
    :class:`~isaaclab.actuators.ImplicitActuator`, execute per group.
    """

    # Initialization.

    def __init__(
        self,
        actuator_cfgs: dict[str, ActuatorBaseCfg],
        control: ActuatorControl,
        *,
        debug_value_resolution: bool = False,
    ):
        """Initialize the actuator collection.

        Args:
            actuator_cfgs: Mapping of actuator group names to actuator configs.
            control: Backend control bridge for state reads and sim writes.
            debug_value_resolution: Whether to log actuator value resolution.
        """
        self._control = control
        self._groups: dict[str, ActuatorBase | object] = {}
        self._group_joint_names: dict[str, list[str]] = {}
        self._group_joint_indices: dict[str, slice | torch.Tensor] = {}
        self._implicit_group_names: set[str] = set()
        self._native_group_names: set[str] = set()
        self._debug_value_resolution = debug_value_resolution
        self._joint_property_resolution_rows: dict[str, dict[str, tuple[tuple[object, ...], ...]]] = {}
        self._has_implicit_actuators = False
        self._launch_cache = _WarpLaunchCache(self.device)

        resolved_cfgs = {name: cfg.copy() for name, cfg in actuator_cfgs.items()}
        resolved_group_joints = self._resolve_group_joints(resolved_cfgs)
        self._allocate_buffers()
        self._target_command = ActuatorTargetCommand(self)
        self._output_command = ActuatorOutputCommand(self)
        for name, cfg in resolved_cfgs.items():
            _resolve_limit_aliases(name, cfg, resolved_group_joints[name][1])
        self._native_group_names = self._control.prepare_native_actuators(self, resolved_cfgs)
        self._build_groups(resolved_cfgs, resolved_group_joints)
        self._newton_selection = self._control.finalize_native_actuators(self)
        if self._native_group_names:
            if self._newton_selection is None:
                raise RuntimeError(
                    "The backend declared Newton-executed actuator groups "
                    f"{sorted(self._native_group_names)} but finalize_native_actuators returned no selection."
                )
            for actuator_name in self._native_group_names:
                self._groups[actuator_name] = self._resolve_newton_group_actuators(actuator_name)
        self._validate_coverage()
        self._build_execution_plan()
        if self._debug_value_resolution:
            self._print_value_resolution_table()
        if not self._control.native_actuator_path_active:
            explicit_group_names = [
                name for name, actuator in self._groups.items() if isinstance(actuator, IdealPDActuator)
            ]
            if explicit_group_names:
                warnings.warn(
                    "Isaac Lab execution of explicit actuator models is deprecated. Use Newton actuator execution "
                    f"instead. Affected groups: {', '.join(explicit_group_names)}.",
                    DeprecationWarning,
                    stacklevel=2,
                )

    # Public interface.

    def __getitem__(self, name: str) -> ActuatorBase | object:
        return self._groups[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __setitem__(self, name: str, actuator: ActuatorBase) -> None:
        raise TypeError("ActuatorCollection membership is fixed after initialization.")

    def __delitem__(self, name: str) -> None:
        raise TypeError("ActuatorCollection membership is fixed after initialization.")

    @property
    def target_command(self) -> ActuatorTargetCommand:
        """Commands received by the actuator models."""
        return self._target_command

    @property
    def output_command(self) -> ActuatorOutputCommand:
        """Processed commands produced for the simulated joints.

        This view is not submitted-command telemetry for native controllers, which
        bypass the processed-command arrays.
        """
        return self._output_command

    @property
    def num_instances(self) -> int:
        """Number of articulation instances."""
        return self._control.num_instances

    @property
    def num_joints(self) -> int:
        """Number of articulation joints."""
        return self._control.num_joints

    @property
    def device(self) -> str:
        """Warp/Torch device string."""
        return self._control.device

    @property
    def has_implicit_actuators(self) -> bool:
        """Whether any configured actuator group is implicit."""
        return self._has_implicit_actuators

    @property
    def computed_effort(self) -> ProxyArray:
        """Joint efforts computed before clipping [N or N·m, depending on joint type]."""
        return self._computed_effort_ta

    @property
    def applied_effort(self) -> ProxyArray:
        """Joint efforts applied after clipping [N or N·m, depending on joint type]."""
        return self._applied_effort_ta

    # Lifecycle.

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset all actuator group states.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
        """
        group_env_ids = self._control._normalize_index_sequence(env_ids)
        for actuator in self._groups.values():
            # Newton-executed groups are reset through the backend below.
            if isinstance(actuator, ActuatorBase):
                actuator.reset(group_env_ids)
        self._control.reset_native_actuators(slice(None) if group_env_ids is None else group_env_ids)

    def compute(self, dt: float = 0.0) -> None:
        """Compute processed actuator commands and telemetry.

        Args:
            dt: Physics step size [s].
        """
        if self._control.compute_native_actuators(self, dt):
            return

        if self._implicit_executor is not None:
            self._implicit_executor.launch(self)
        joint_pos = self._control.joint_pos
        joint_vel = self._control.joint_vel
        for actuator, joint_indices_wp in self._execution_actuators:
            joint_indices = actuator.joint_indices
            control_action = ArticulationActions(
                joint_positions=self.target_command.position.torch[:, joint_indices],
                joint_velocities=self.target_command.velocity.torch[:, joint_indices],
                joint_efforts=self.target_command.effort.torch[:, joint_indices],
                joint_indices=joint_indices,
            )
            control_action = actuator.compute(
                control_action,
                joint_pos=joint_pos.torch[:, joint_indices],
                joint_vel=joint_vel.torch[:, joint_indices],
            )
            self._scatter_actuator_output(actuator, control_action, joint_indices_wp)

    def submit_commands(self) -> None:
        """Submit processed actuator command buffers through the backend control object."""
        self._control.submit_commands(self)

    def _newton_group_columns(self, name: str) -> torch.Tensor:
        """Backend view columns of one group's joints, in group joint order."""
        joint_ids = self._group_joint_indices[name]
        if isinstance(joint_ids, slice):
            columns = torch.arange(self.num_joints, device=self.device)
        else:
            columns = joint_ids.to(self.device, dtype=torch.long)
        user_to_backend = self._newton_selection.joint_user_to_backend_indices
        if user_to_backend is not None:
            columns = torch.tensor(user_to_backend, dtype=torch.long, device=self.device)[columns]
        return columns

    # Construction and property resolution.

    def _allocate_buffers(self) -> None:
        """Allocate articulation-wide command and telemetry buffers."""
        shape = (self.num_instances, self.num_joints)
        # Staging buffers for the actuators I/O
        self._joint_pos_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_vel_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_effort_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_pos_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_vel_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_effort_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        # Telemetry buffers
        self._computed_effort = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._applied_effort = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._soft_joint_vel_limits = wp.zeros(shape, dtype=wp.float32, device=self.device)
        # All joint IDs and masks
        self._all_joint_ids = wp.array(list(range(self.num_joints)), dtype=wp.int32, device=self.device)
        self._all_true_env_mask = wp.ones(self.num_instances, dtype=wp.bool, device=self.device)
        self._all_true_joint_mask = wp.ones(self.num_joints, dtype=wp.bool, device=self.device)

        # Proxy arrays for the buffers
        self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
        self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
        self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
        self._joint_pos_target_sim_ta = ProxyArray(self._joint_pos_target_sim)
        self._joint_vel_target_sim_ta = ProxyArray(self._joint_vel_target_sim)
        self._joint_effort_target_sim_ta = ProxyArray(self._joint_effort_target_sim)
        self._computed_effort_ta = ProxyArray(self._computed_effort)
        self._applied_effort_ta = ProxyArray(self._applied_effort)

    def _resolve_group_joints(
        self, actuator_cfgs: dict[str, ActuatorBaseCfg]
    ) -> dict[str, tuple[ProxyArray, list[str]]]:
        """Resolve group selectors and reject joints assigned to multiple groups."""
        resolved: dict[str, tuple[ProxyArray, list[str]]] = {}
        joint_owners: dict[str, str] = {}
        # Resolve exact joint names and indices
        for actuator_name, actuator_cfg in actuator_cfgs.items():
            joint_ids, joint_names = self._control.find_joints(actuator_cfg.joint_names_expr)
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            # Check for duplicate joint names
            for joint_name in joint_names:
                owner = joint_owners.get(joint_name)
                if owner is not None and owner != actuator_name:
                    raise ValueError(
                        f"Joint '{joint_name}' is assigned to multiple actuator groups: '{owner}' and"
                        f" '{actuator_name}'."
                    )
                joint_owners[joint_name] = actuator_name
            resolved[actuator_name] = (joint_ids, joint_names)
        return resolved

    def _build_groups(
        self,
        actuator_cfgs: dict[str, ActuatorBaseCfg],
        resolved_group_joints: dict[str, tuple[list[int] | ProxyArray, list[str]]],
    ) -> None:
        """Construct actuator groups and apply their resolved joint properties.

        Newton-executed groups never instantiate an Isaac Lab actuator model: the
        Newton controllers own their parameters, so a Lab model would only hold
        misleading construction-time snapshots. Their mapping entries are filled
        with the owning Newton actuator objects after backend finalization.
        """
        construction_records: list[tuple[dict[str, torch.Tensor], torch.Tensor | slice, bool, bool]] = []
        for actuator_name, actuator_cfg in actuator_cfgs.items():
            joint_ids, joint_names = resolved_group_joints[actuator_name]
            if len(joint_names) == self.num_joints:
                actuator_joint_ids: slice | torch.Tensor = slice(None)
            elif isinstance(joint_ids, ProxyArray):
                actuator_joint_ids = joint_ids.torch
            else:
                actuator_joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.int32)
            self._group_joint_names[actuator_name] = joint_names
            self._group_joint_indices[actuator_name] = actuator_joint_ids

            joint_defaults = self._control.get_default_joint_properties(actuator_joint_ids)
            implicit = _is_implicit_actuator_cfg(actuator_cfg)
            self._has_implicit_actuators = self._has_implicit_actuators or implicit
            if implicit:
                self._implicit_group_names.add(actuator_name)
            native_managed = actuator_name in self._native_group_names
            properties, table_rows = self._resolve_joint_properties(
                actuator_cfg,
                joint_defaults,
                joint_names,
                actuator_joint_ids,
            )
            if native_managed:
                # placeholder keeps configuration order; replaced by the Newton actuator
                # objects once the backend selection is finalized.
                self._groups[actuator_name] = None
            else:
                actuator_kwargs = dict(
                    cfg=actuator_cfg,
                    joint_names=joint_names,
                    joint_ids=actuator_joint_ids,
                    num_envs=self.num_instances,
                    device=self.device,
                    stiffness=properties["stiffness"],
                    damping=properties["damping"],
                    actuator_velocity_limit=properties["joint_velocity_limit"],
                )
                if implicit:
                    # implicit groups read the solver limit live after binding; the resolved value
                    # seeds the pre-binding construction buffer.
                    actuator_kwargs["joint_effort_limit"] = properties["joint_effort_limit"]
                else:
                    # explicit models default their clip limit to the authored joint effort limit.
                    actuator_kwargs["actuator_effort_limit"] = joint_defaults["joint_effort_limit"]
                self._groups[actuator_name] = actuator_cfg.class_type(**actuator_kwargs)
            if self._debug_value_resolution:
                self._joint_property_resolution_rows[actuator_name] = table_rows
            construction_records.append(
                (
                    properties,
                    actuator_joint_ids,
                    implicit,
                    native_managed,
                )
            )

        for properties, joint_ids, implicit, native_managed in construction_records:
            self._control.write_resolved_joint_properties(
                properties,
                joint_ids,
                implicit=implicit,
                native_managed=native_managed,
            )
        for actuator in self._groups.values():
            if isinstance(actuator, ImplicitActuator):
                actuator._bind_actuator_parameters(self._control)

    def _resolve_newton_group_actuators(self, name: str) -> object:
        """Return the Newton actuator object(s) that drive one group's joints.

        Newton merges structurally identical joints into one actuator, so the
        returned object can be shared between groups. A single covering actuator
        is returned directly; a group spanning several returns them as a tuple.
        """
        view = self._newton_selection.view
        columns = self._newton_group_columns(name)
        matched = []
        for actuator in self._newton_selection.actuators:
            mapping = wp.to_torch(view._get_actuator_dof_mapping(actuator))
            env_columns = mapping.reshape(self.num_instances, -1)[0]
            if bool((env_columns[columns] >= 0).any()):
                matched.append(actuator)
        if not matched:
            raise RuntimeError(f"No Newton actuator drives any joint of group '{name}'.")
        return matched[0] if len(matched) == 1 else tuple(matched)

    def _implicit_group_joint_indices(self) -> list[slice | torch.Tensor]:
        """Joint selectors of the implicit groups, consumed by backend implicit-DOF masks."""
        return [self._group_joint_indices[name] for name in self._implicit_group_names]

    def _resolve_joint_properties(
        self,
        cfg: ActuatorBaseCfg,
        defaults: dict[str, torch.Tensor],
        joint_names: list[str],
        joint_ids: torch.Tensor | slice,
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[tuple[object, ...], ...]]]:
        """Resolve fresh construction-only joint properties for one actuator group.

        The solver keeps the authored joint limits unless the configuration overrides
        them; explicit actuator models no longer widen the solver effort limit.
        """
        values: dict[str, torch.Tensor] = {}
        resolution_rows: dict[str, tuple[tuple[object, ...], ...]] = {}
        for cfg_name in (
            "stiffness",
            "damping",
            "armature",
            "friction",
            "dynamic_friction",
            "viscous_friction",
            "joint_effort_limit",
            "joint_velocity_limit",
        ):
            default_value = defaults[cfg_name]
            cfg_value = getattr(cfg, cfg_name)
            value = self._resolve_joint_property(cfg_value, default_value, joint_names)
            values[cfg_name] = value
            if self._debug_value_resolution:
                rows = self._joint_property_resolution_rows_for(
                    cfg_value,
                    value,
                    default_value,
                    joint_names,
                    joint_ids,
                )
                if rows:
                    resolution_rows[cfg_name] = rows

        return values, resolution_rows

    def _resolve_joint_property(
        self,
        cfg_value: float | dict[str, float] | None,
        default_value: torch.Tensor,
        joint_names: list[str],
    ) -> torch.Tensor:
        """Resolve one group-shaped joint property from config and authored defaults."""
        return resolve_joint_parameter(cfg_value, default_value, joint_names, self.num_instances, self.device)

    def _joint_property_resolution_rows_for(
        self,
        cfg_value: float | dict[str, float] | None,
        value: torch.Tensor,
        default_value: torch.Tensor,
        joint_names: list[str],
        joint_ids: torch.Tensor | slice,
    ) -> tuple[tuple[object, ...], ...]:
        """Formats joint property resolution rows for debugging. (Actuators property table output.)"""
        if cfg_value is not None and torch.allclose(value, default_value):
            return ()
        if isinstance(joint_ids, slice):
            ids = range(self.num_joints)
        else:
            ids = tuple(int(joint_id) for joint_id in joint_ids.tolist())
        return tuple(
            (
                name,
                ids[index],
                float(default_value[0, index]),
                "Not Specified" if cfg_value is None else float(value[0, index]),
                float(default_value[0, index]) if cfg_value is None else float(value[0, index]),
            )
            for index, name in enumerate(joint_names)
        )

    # Execution planning and runtime.

    def _joint_indices_as_wp(self, actuator: ActuatorBase) -> wp.array(dtype=wp.int32):
        """Return an actuator group's joint indices as a Warp int32 array."""
        if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
            return self._all_joint_ids
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, wp.array):
            return joint_indices
        return wp.from_torch(joint_indices.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)

    def _joint_indices_as_torch(self, actuator: ActuatorBase) -> torch.Tensor:
        """Return an actuator group's joint indices as a contiguous Torch int32 tensor."""
        if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
            return torch.arange(self.num_joints, dtype=torch.int32, device=self.device)
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, wp.array):
            joint_indices = wp.to_torch(joint_indices)
        return joint_indices.to(self.device, dtype=torch.int32).contiguous()

    def _build_execution_plan(self) -> None:
        """Partition the actuator groups into the fused implicit executor and a per-group list.

        Plain :class:`~isaaclab.actuators.ImplicitActuator` groups only produce effort
        telemetry, so they are not executed one group at a time: their joint indices are
        aggregated while parsing and one executor computes all of them in a single fused
        kernel launch. Subclasses may override :meth:`~ActuatorBase.compute`, so they stay
        in the per-group list along with the explicit models. On a native actuator path the
        backend replaces :meth:`compute` entirely and the plan is left empty.
        """
        self._execution_actuators: list[tuple[ActuatorBase, wp.array(dtype=wp.int32)]] = []
        implicit_names: list[str] = []
        implicit_groups: list[ImplicitActuator] = []
        if self._control.native_actuator_path_active:
            self._implicit_executor = None
            return
        for name, group in self._groups.items():
            if name in self._native_group_names:
                continue
            if type(group) is ImplicitActuator:
                implicit_names.append(name)
                implicit_groups.append(group)
            else:
                self._execution_actuators.append((group, self._joint_indices_as_wp(group)))
        self._implicit_executor = (
            _ImplicitExecutor(self, tuple(implicit_names), tuple(implicit_groups)) if implicit_groups else None
        )

    def _rebind_state_inputs(self) -> None:
        """Rebind the implicit executor after backend state storage is replaced.

        Per-group execution reads backend state through the control object on every
        :meth:`compute` call, so only the cached implicit launch holds state references
        that need rebinding.
        """
        if self._implicit_executor is not None:
            self._implicit_executor.rebind(self)

    def _scatter_actuator_output(
        self,
        actuator: ActuatorBase,
        control_action: ArticulationActions,
        joint_indices: wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Publish one explicit actuator's processed commands and telemetry."""
        if joint_indices is None:
            joint_indices = self._joint_indices_as_wp(actuator)
        inputs = [
            control_action.joint_positions,
            control_action.joint_velocities,
            control_action.joint_efforts,
            actuator.computed_effort,
            actuator.applied_effort,
            actuator.actuator_velocity_limit,
            joint_indices,
        ]
        outputs = [
            self._joint_pos_target_sim,
            self._joint_vel_target_sim,
            self._joint_effort_target_sim,
            self._computed_effort,
            self._applied_effort,
            self._soft_joint_vel_limits,
        ]
        wp.launch(
            actuator_kernels.scatter_explicit_actuator_outputs,
            dim=(self.num_instances, joint_indices.shape[0]),
            inputs=inputs,
            outputs=outputs,
            device=self.device,
        )

    # Diagnostics.

    def _validate_coverage(self) -> None:
        """Warn when actuator groups do not cover the expected movable joints."""
        if self.num_joints == 0:
            return
        total_act_joints = sum(len(joint_names) for joint_names in self._group_joint_names.values())
        expected_joints = self.num_joints - self._control.num_fixed_tendons
        if total_act_joints != expected_joints:
            logger.warning(
                "Actuator groups cover %s joints; expected %s after accounting for fixed tendons.",
                total_act_joints,
                expected_joints,
            )

    def _print_value_resolution_table(self) -> None:
        """Log construction-time differences between authored and configured values."""
        table = PrettyTable(["Group", "Property", "Name", "ID", "USD Value", "ActuatorCfg Value", "Applied"])
        for actuator_group in self._groups:
            group_count = 0
            for property_name, resolution_details in self._joint_property_resolution_rows[actuator_group].items():
                for prop_idx, resolution_detail in enumerate(resolution_details):
                    actuator_group_str = actuator_group if group_count == 0 else ""
                    property_str = property_name if prop_idx == 0 else ""
                    fmt = [f"{value:.2e}" if isinstance(value, float) else str(value) for value in resolution_detail]
                    table.add_row([actuator_group_str, property_str, *fmt])
                    group_count += 1
        logger.warning("\nActuatorCfg-USD Value Discrepancy Resolution (matching values are skipped): \n%s", table)


class _ImplicitExecutor:
    """Fused executor for the collection's plain :class:`ImplicitActuator` groups.

    A single group executes as itself. Multiple groups execute through one private
    shadow actuator covering the union of their joint indices, so the per-step cost
    is one kernel launch regardless of how the configuration partitions the joints.
    The logical groups' telemetry tensors are re-pointed at contiguous views of the
    shadow's buffers, so per-group reads observe the fused results directly.
    """

    _cache_key_counter = itertools.count()
    """Monotonic launch-cache key source; ``id()`` keys could be reused after garbage collection."""

    def __init__(self, collection: ActuatorCollection, names: tuple[str, ...], groups: tuple[ImplicitActuator, ...]):
        self._cache_key = ("implicit", next(_ImplicitExecutor._cache_key_counter))
        self.group_names = names
        if len(groups) == 1:
            self.actuator = groups[0]
            joint_indices = collection._joint_indices_as_torch(groups[0])
        else:
            joint_indices = torch.cat([collection._joint_indices_as_torch(group) for group in groups])
            self.actuator = self._build_shadow_actuator(groups, joint_indices)
        self.joint_indices_wp = wp.from_torch(joint_indices, dtype=wp.int32)
        self.kernel_inputs: list[wp.array] | None = None
        self.kernel_outputs: list[wp.array] | None = None
        self._assemble_kernel_arrays(collection)

    @staticmethod
    def _build_shadow_actuator(groups: tuple[ImplicitActuator, ...], joint_indices: torch.Tensor) -> ImplicitActuator:
        """Build one private shadow actuator covering all the logical groups' joints.

        Retains the first group's config metadata; replaces the execution tensors below
        without cloning the logical groups' tensor storage. The groups' telemetry tensors
        become contiguous views of the shadow's buffers.
        """
        shadow = copy.copy(groups[0])
        shadow._joint_names = [name for group in groups for name in group.joint_names]
        shadow._joint_indices = joint_indices
        shadow.actuator_velocity_limit = torch.cat([group.actuator_velocity_limit for group in groups], dim=1)
        shadow.computed_effort = torch.zeros(shadow._num_envs, len(shadow._joint_names), device=shadow._device)
        shadow.applied_effort = torch.zeros_like(shadow.computed_effort)
        start = 0
        for group in groups:
            group_slice = slice(start, start + group.num_joints)
            start += group.num_joints
            group.computed_effort = shadow.computed_effort[:, group_slice]
            group.applied_effort = shadow.applied_effort[:, group_slice]
        return shadow

    def _assemble_kernel_arrays(self, collection: ActuatorCollection) -> None:
        """Assemble the implicit kernel argument arrays.

        Existing argument lists are updated in place so holders of the list objects
        observe rebound backend state.
        """
        control = collection._control
        inputs = [
            collection._joint_pos_target,
            collection._joint_vel_target,
            collection._joint_effort_target,
            control.joint_pos.warp,
            control.joint_vel.warp,
            control.joint_stiffness.warp,
            control.joint_damping.warp,
            control.joint_effort_limits.warp,
            wp.from_torch(self.actuator.actuator_velocity_limit, dtype=wp.float32),
            self.joint_indices_wp,
        ]
        outputs = [
            wp.from_torch(self.actuator.computed_effort, dtype=wp.float32),
            wp.from_torch(self.actuator.applied_effort, dtype=wp.float32),
            collection._joint_pos_target_sim,
            collection._joint_vel_target_sim,
            collection._joint_effort_target_sim,
            collection._computed_effort,
            collection._applied_effort,
            collection._soft_joint_vel_limits,
        ]
        if self.kernel_inputs is None:
            self.kernel_inputs = inputs
            self.kernel_outputs = outputs
        else:
            self.kernel_inputs[:] = inputs
            self.kernel_outputs[:] = outputs

    def launch(self, collection: ActuatorCollection) -> None:
        """Compute all the executor's joints through the cached Warp launch."""
        collection._launch_cache.launch(
            self._cache_key,
            actuator_kernels.compute_implicit_actuator_batch,
            dim=(collection.num_instances, self.joint_indices_wp.shape[0]),
            inputs=self.kernel_inputs,
            outputs=self.kernel_outputs,
        )

    def rebind(self, collection: ActuatorCollection) -> None:
        """Reassemble the kernel arguments after backend state storage is replaced."""
        self._assemble_kernel_arrays(collection)
        collection._launch_cache.clear(self._cache_key)


class ActuatorTargetCommand:
    """Commands received by the actuator models.

    Position and velocity commands use joint-side coordinates. All command
    arrays are indexed by articulation joint, not by motor shaft.

    Index selectors must contain unique environment and joint indices. Repeated
    indices dispatch concurrent writes to the same destination and produce an
    undefined result. Deduplicate selectors or use mask setters.
    """

    def __init__(self, collection: ActuatorCollection) -> None:
        """Initialize the command view.

        Args:
            collection: Owning actuator collection.
        """
        self._collection = collection

    @property
    def position(self) -> ProxyArray:
        """Desired positions [m or rad, depending on joint type]."""
        return self._collection._joint_pos_target_ta

    @property
    def velocity(self) -> ProxyArray:
        """Desired velocities [m/s or rad/s, depending on joint type]."""
        return self._collection._joint_vel_target_ta

    @property
    def effort(self) -> ProxyArray:
        """Effort commands [N or N·m, depending on joint type]."""
        return self._collection._joint_effort_target_ta

    def set_position_index(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set desired positions using indices.

        Args:
            value: Desired positions [m or rad, depending on joint type]. Shape is
                ``(len(env_ids), len(joint_ids))``, or ``(num_instances, num_joints)`` when
                :paramref:`full_data` is true.
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether :paramref:`value` is a full articulation command buffer.
        """
        collection = self._collection
        env_ids_resolved = collection._control.resolve_env_ids(env_ids)
        joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            value,
            env_ids_resolved,
            joint_ids_resolved,
            collection._joint_pos_target,
            full_data=full_data,
            command_name="position",
        )

    def set_velocity_index(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set desired velocities using indices.

        Args:
            value: Desired velocities [m/s or rad/s, depending on joint type]. Shape is
                ``(len(env_ids), len(joint_ids))``, or ``(num_instances, num_joints)`` when
                :paramref:`full_data` is true.
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether :paramref:`value` is a full articulation command buffer.
        """
        collection = self._collection
        env_ids_resolved = collection._control.resolve_env_ids(env_ids)
        joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            value,
            env_ids_resolved,
            joint_ids_resolved,
            collection._joint_vel_target,
            full_data=full_data,
            command_name="velocity",
        )

    def set_effort_index(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set effort commands using indices.

        Args:
            value: Effort commands [N or N·m, depending on joint type]. Shape is
                ``(len(env_ids), len(joint_ids))``, or ``(num_instances, num_joints)`` when
                :paramref:`full_data` is true.
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether :paramref:`value` is a full articulation command buffer.
        """
        collection = self._collection
        env_ids_resolved = collection._control.resolve_env_ids(env_ids)
        joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            value,
            env_ids_resolved,
            joint_ids_resolved,
            collection._joint_effort_target,
            full_data=full_data,
            command_name="effort",
        )

    def set_position_mask(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_mask: wp.array(dtype=wp.bool) | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set desired positions using masks.

        Args:
            value: Full articulation position commands [m or rad, depending on joint type]. Shape is
                ``(num_instances, num_joints)``.
            joint_mask: Joint selection mask. Defaults to all joints.
            env_mask: Environment selection mask. Defaults to all environments.
        """
        collection = self._collection
        env_mask_resolved = self._resolve_mask(env_mask, collection._all_true_env_mask, "env_mask")
        joint_mask_resolved = self._resolve_mask(joint_mask, collection._all_true_joint_mask, "joint_mask")
        self._write_mask_target(
            value,
            env_mask_resolved,
            joint_mask_resolved,
            collection._joint_pos_target,
            command_name="position",
        )

    def set_velocity_mask(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_mask: wp.array(dtype=wp.bool) | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set desired velocities using masks.

        Args:
            value: Full articulation velocity commands [m/s or rad/s, depending on joint type]. Shape is
                ``(num_instances, num_joints)``.
            joint_mask: Joint selection mask. Defaults to all joints.
            env_mask: Environment selection mask. Defaults to all environments.
        """
        collection = self._collection
        env_mask_resolved = self._resolve_mask(env_mask, collection._all_true_env_mask, "env_mask")
        joint_mask_resolved = self._resolve_mask(joint_mask, collection._all_true_joint_mask, "joint_mask")
        self._write_mask_target(
            value,
            env_mask_resolved,
            joint_mask_resolved,
            collection._joint_vel_target,
            command_name="velocity",
        )

    def set_effort_mask(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_mask: wp.array(dtype=wp.bool) | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set effort commands using masks.

        Args:
            value: Full articulation effort commands [N or N·m, depending on joint type]. Shape is
                ``(num_instances, num_joints)``.
            joint_mask: Joint selection mask. Defaults to all joints.
            env_mask: Environment selection mask. Defaults to all environments.
        """
        collection = self._collection
        env_mask_resolved = self._resolve_mask(env_mask, collection._all_true_env_mask, "env_mask")
        joint_mask_resolved = self._resolve_mask(joint_mask, collection._all_true_joint_mask, "joint_mask")
        self._write_mask_target(
            value,
            env_mask_resolved,
            joint_mask_resolved,
            collection._joint_effort_target,
            command_name="effort",
        )

    @staticmethod
    def _resolve_mask(
        mask: wp.array(dtype=wp.bool) | None, all_true_mask: wp.array(dtype=wp.bool), name: str
    ) -> wp.array(dtype=wp.bool):
        """Return the full selection mask for an optional ``wp.bool`` mask argument."""
        if mask is None:
            return all_true_mask
        if not isinstance(mask, wp.array) or mask.dtype != wp.bool:
            raise TypeError(f"Expected '{name}' to be a wp.array of dtype wp.bool, got {type(mask)!r}.")
        return mask

    def _write_index_target(
        self,
        target: torch.Tensor | wp.array(dtype=wp.float32),
        env_ids: torch.Tensor | wp.array,
        joint_ids: torch.Tensor | wp.array,
        target_buffer: wp.array(dtype=wp.float32),
        *,
        full_data: bool,
        command_name: str,
    ) -> None:
        collection = self._collection
        expected_shape = (
            (collection.num_instances, collection.num_joints) if full_data else (env_ids.shape[0], joint_ids.shape[0])
        )
        collection._control.assert_shape_and_dtype(target, expected_shape, wp.float32, "target")
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(env_ids, joint_ids),
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[target, env_ids, joint_ids, full_data],
            outputs=[target_buffer],
            device=collection.device,
        )
        collection._control.stage_user_command(command_name, collection, env_ids, joint_ids, None, None)

    def _write_mask_target(
        self,
        target: torch.Tensor | wp.array(dtype=wp.float32),
        env_mask: wp.array(dtype=wp.bool),
        joint_mask: wp.array(dtype=wp.bool),
        target_buffer: wp.array(dtype=wp.float32),
        *,
        command_name: str,
    ) -> None:
        collection = self._collection
        collection._control.assert_shape_and_dtype_mask(target, (env_mask, joint_mask), wp.float32, "target")
        wp.launch(
            actuator_kernels.write_2d_float_with_mask,
            dim=(env_mask.shape[0], joint_mask.shape[0]),
            inputs=[target, env_mask, joint_mask],
            outputs=[target_buffer],
            device=collection.device,
        )
        collection._control.stage_user_command(command_name, collection, None, None, env_mask, joint_mask)


class ActuatorOutputCommand:
    """Processed commands produced for the simulated joints.

    These arrays contain submitted-command telemetry for Isaac Lab-managed
    actuator models. Native controllers bypass the arrays, so they do not
    provide submitted-command telemetry on a native path.
    """

    def __init__(self, collection: ActuatorCollection) -> None:
        """Initialize the joint command view.

        Args:
            collection: Owning actuator collection.
        """
        self._collection = collection

    @property
    def position(self) -> ProxyArray:
        """Processed position commands [m or rad, depending on joint type]."""
        return self._collection._joint_pos_target_sim_ta

    @property
    def velocity(self) -> ProxyArray:
        """Processed velocity commands [m/s or rad/s, depending on joint type]."""
        return self._collection._joint_vel_target_sim_ta

    @property
    def effort(self) -> ProxyArray:
        """Processed effort commands [N or N·m, depending on joint type]."""
        return self._collection._joint_effort_target_sim_ta
