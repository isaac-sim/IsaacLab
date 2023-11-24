# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import carb
import omni.physics.tensors.impl.api as physx
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.types import ArticulationActions
from pxr import Usd, UsdPhysics

import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.actuators import ActuatorBase, ActuatorBaseCfg, ImplicitActuator

from ..rigid_object import RigidObject
from .articulation_data import ArticulationData

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg


class Articulation(RigidObject):
    """Class for handling articulations."""

    cfg: ArticulationCfg
    """Configuration instance for the articulations."""

    def __init__(self, cfg: ArticulationCfg):
        """Initialize the articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        # container for data access
        self._data = ArticulationData()
        # data for storing actuator group
        self.actuators: dict[str, ActuatorBase] = dict.fromkeys(self.cfg.actuators.keys())

    """
    Properties
    """

    @property
    def data(self) -> ArticulationData:
        """Data related to articulation."""
        return self._data

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        return self._is_fixed_base

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self.root_view.num_dof

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self.root_view.num_bodies

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self.root_view.dof_names

    @property
    def root_view(self) -> ArticulationView:
        return self._root_view

    @property
    def body_view(self) -> RigidPrimView:
        return self._body_view

    @property
    def root_physx_view(self) -> physx.ArticulationView:
        return self._root_view._physics_view  # pyright: ignore [reportPrivateUsage]

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        return self._body_view._physics_view  # pyright: ignore [reportPrivateUsage]

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = slice(None)
        # reset actuators
        for actuator in self.actuators.values():
            actuator.reset(env_ids)

    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.
        """
        super().write_data_to_sim()
        # write commands
        self._apply_actuator_model()
        # apply actions into simulation
        self.root_physx_view.set_dof_actuation_forces(self._joint_effort_target_sim, self._ALL_INDICES)
        # position and velocity targets only for implicit actuators
        if self._has_implicit_actuators:
            self.root_physx_view.set_dof_position_targets(self._joint_pos_target_sim, self._ALL_INDICES)
            self.root_physx_view.set_dof_velocity_targets(self._joint_vel_target_sim, self._ALL_INDICES)

    def update(self, dt: float | None = None):
        # -- root state (note: we roll the quaternion to match the convention used in Isaac Sim -- wxyz)
        self._data.root_state_w[:, :7] = self.root_physx_view.get_root_transforms()
        self._data.root_state_w[:, 3:7] = math_utils.convert_quat(self._data.root_state_w[:, 3:7], to="wxyz")
        self._data.root_state_w[:, 7:] = self.root_physx_view.get_root_velocities()
        # -- joint states
        self._data.joint_pos[:] = self.root_physx_view.get_dof_positions()
        self._data.joint_vel[:] = self.root_physx_view.get_dof_velocities()
        self._data.joint_acc[:] = (self._data.joint_vel - self._previous_joint_vel) / dt
        # -- update common data
        # note: these are computed in the base class
        self._update_common_data(dt)
        # -- update history buffers
        self._previous_joint_vel[:] = self._data.joint_vel[:]

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None
    ) -> tuple[list[int], list[str]]:
        """Find joints in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self.joint_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset)

    """
    Operations - Writers.
    """

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, :7] = root_pose.clone()
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_state_w[:, :7].clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_root_transforms(root_poses_xyzw, indices=env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, 7:] = root_velocity.clone()
        # set into simulation
        self.root_physx_view.set_root_velocities(self._data.root_state_w[:, 7:], indices=env_ids)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint positions and velocities to the simulation.

        Args:
            position: Joint positions. Shape is ``(len(env_ids), len(joint_ids))``.
            velocity: Joint velocities. Shape is ``(len(env_ids), len(joint_ids))``.
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # set into internal buffers
        self._data.joint_pos[env_ids, joint_ids] = position
        self._data.joint_vel[env_ids, joint_ids] = velocity
        self._previous_joint_vel[env_ids, joint_ids] = velocity
        self._data.joint_acc[env_ids, joint_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_dof_positions(self._data.joint_pos, indices=env_ids)
        self.root_physx_view.set_dof_velocities(self._data.joint_vel, indices=env_ids)

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is ``(len(env_ids), len(joint_ids))``.
            joint_ids: The joint indices to set the stiffness for. Defaults to None (all joints).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # set into internal buffers
        self._data.joint_stiffness[env_ids, joint_ids] = stiffness
        # set into simulation
        self.root_physx_view.set_dof_stiffnesses(self._data.joint_stiffness.cpu(), indices=env_ids.cpu())

    def write_joint_damping_to_sim(
        self, damping: torch.Tensor, joint_ids: Sequence[int] | None = None, env_ids: Sequence[int] | None = None
    ):
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is ``(len(env_ids), len(joint_ids))``.
            joint_ids: The joint indices to set the damping for.
                Defaults to None (all joints).
            env_ids: The environment indices to set the damping for.
                Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # set into internal buffers
        self._data.joint_damping[env_ids, joint_ids] = damping
        # set into simulation
        self.root_physx_view.set_dof_dampings(self._data.joint_damping.cpu(), indices=env_ids.cpu())

    def write_joint_torque_limit_to_sim(
        self,
        limits: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint torque limits into the simulation.

        Args:
            limits: Joint torque limits. Shape is ``(len(env_ids), len(joint_ids))``.
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # set into internal buffers
        torque_limit_all = self.root_physx_view.get_dof_max_forces()
        torque_limit_all[env_ids, joint_ids] = limits
        # set into simulation
        self.root_physx_view.set_dof_max_forces(torque_limit_all.cpu(), indices=env_ids.cpu())

    """
    Operations - State.
    """

    def set_joint_position_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint position targets into internal buffers.

        .. note::
            This function does not apply the joint targets to the simulation. It only fills the buffers with
            the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is ``(len(env_ids), len(joint_ids))``.
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # set targets
        self._data.joint_pos_target[env_ids, joint_ids] = target

    def set_joint_velocity_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint velocity targets into internal buffers.

        .. note::
            This function does not apply the joint targets to the simulation. It only fills the buffers with
            the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is ``(len(env_ids), len(joint_ids))``.
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # set targets
        self._data.joint_vel_target[env_ids, joint_ids] = target

    def set_joint_effort_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint efforts into internal buffers.

        .. note::
            This function does not apply the joint targets to the simulation. It only fills the buffers with
            the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is ``(len(env_ids), len(joint_ids))``.
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # set targets
        self._data.joint_effort_target[env_ids, joint_ids] = target

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # -- articulation
        self._root_view = ArticulationView(self.cfg.prim_path, reset_xform_properties=False)
        # Hacking the initialization of the articulation view.
        # reason: The default initialization of the articulation view is not working properly as it tries to create
        # default actions that is not possible within the post-play callback.
        # We override their internal function that throws an error which is not desired or needed.
        dummy_tensor = torch.empty(size=(0, 0), device=self.device)
        dummy_joint_actions = ArticulationActions(dummy_tensor, dummy_tensor, dummy_tensor)
        current_fn = self._root_view.get_applied_actions
        self._root_view.get_applied_actions = lambda *args, **kwargs: dummy_joint_actions
        # initialize the root view
        self._root_view.initialize()
        # restore the function
        self._root_view.get_applied_actions = current_fn

        # -- link views
        # note: we use the root view to get the body names, but we use the body view to get the
        #       actual data. This is mainly needed to apply external forces to the bodies.
        body_names_regex = r"(" + "|".join(self.root_view.body_names) + r")"
        body_names_regex = f"{self.cfg.prim_path}/{body_names_regex}"
        self._body_view = RigidPrimView(body_names_regex, reset_xform_properties=False)
        self._body_view.initialize()
        # check that initialization was successful
        if len(self.body_names) != self.num_bodies:
            raise RuntimeError("Failed to initialize all bodies properly in the articulation.")
        # -- fixed base based on root joint
        self._is_fixed_base = False
        for prim in Usd.PrimRange(self._root_view.prims[0]):
            joint_prim = UsdPhysics.FixedJoint(prim)
            # we check all joints under the root prim and classify the asset as fixed base if there exists
            # a fixed joint that has only one target (i.e. the root link).
            if joint_prim and joint_prim.GetJointEnabledAttr().Get():
                body_0_exist = joint_prim.GetBody0Rel().GetTargets() != []
                body_1_exist = joint_prim.GetBody1Rel().GetTargets() != []
                if not (body_0_exist and body_1_exist):
                    self._is_fixed_base = True
                    break
        # log information about the articulation
        carb.log_info(f"Articulation initialized at: {self.cfg.prim_path}")
        carb.log_info(f"Root name: {self.body_names[0]}")
        carb.log_info(f"Is fixed root: {self.is_fixed_base}")
        carb.log_info(f"Number of bodies: {self.num_bodies}")
        carb.log_info(f"Body names: {self.body_names}")
        carb.log_info(f"Number of joints: {self.num_joints}")
        carb.log_info(f"Joint names: {self.joint_names}")
        # -- assert that parsing was successful
        if set(self.root_view.body_names) != set(self.body_names):
            raise RuntimeError("Failed to parse all bodies properly in the articulation.")
        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        self._process_actuators_cfg()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # allocate buffers
        super()._create_buffers()
        # history buffers
        self._previous_joint_vel = torch.zeros(self.root_view.count, self.num_joints, device=self.device)

        # asset data
        # -- properties
        self._data.joint_names = self.joint_names
        # -- joint states
        self._data.joint_pos = torch.zeros(self.root_view.count, self.num_joints, dtype=torch.float, device=self.device)
        self._data.joint_vel = torch.zeros_like(self._data.joint_pos)
        self._data.joint_acc = torch.zeros_like(self._data.joint_pos)
        self._data.default_joint_pos = torch.zeros_like(self._data.joint_pos)
        self._data.default_joint_vel = torch.zeros_like(self._data.joint_pos)
        # -- joint commands
        self._data.joint_pos_target = torch.zeros_like(self._data.joint_pos)
        self._data.joint_vel_target = torch.zeros_like(self._data.joint_pos)
        self._data.joint_effort_target = torch.zeros_like(self._data.joint_pos)
        self._data.joint_stiffness = torch.zeros_like(self._data.joint_pos)
        self._data.joint_damping = torch.zeros_like(self._data.joint_pos)
        # -- joint commands (explicit)
        self._data.computed_torque = torch.zeros_like(self._data.joint_pos)
        self._data.applied_torque = torch.zeros_like(self._data.joint_pos)
        # -- other data
        self._data.soft_joint_pos_limits = torch.zeros(self.root_view.count, self.num_joints, 2, device=self.device)
        self._data.soft_joint_vel_limits = torch.zeros(self.root_view.count, self.num_joints, device=self.device)
        self._data.gear_ratio = torch.ones(self.root_view.count, self.num_joints, device=self.device)

        # soft joint position limits (recommended not to be too close to limits).
        joint_pos_limits = self.root_physx_view.get_dof_limits()
        joint_pos_mean = (joint_pos_limits[..., 0] + joint_pos_limits[..., 1]) / 2
        joint_pos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]
        soft_limit_factor = self.cfg.soft_joint_pos_limit_factor
        # add to data
        self._data.soft_joint_pos_limits[..., 0] = joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
        self._data.soft_joint_pos_limits[..., 1] = joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor

        # create buffers to store processed actions from actuator models
        self._joint_pos_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_vel_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_effort_target_sim = torch.zeros_like(self._data.joint_pos_target)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        super()._process_cfg()
        # -- joint state
        # joint pos
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_pos, self.joint_names
        )
        self._data.default_joint_pos[:, indices_list] = torch.tensor(values_list, device=self.device)
        # joint vel
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_vel, self.joint_names
        )
        self._data.default_joint_vel[:, indices_list] = torch.tensor(values_list, device=self.device)

    """
    Internal helpers -- Actuators.
    """

    def _process_actuators_cfg(self):
        """Process and apply articulation joint properties."""
        # flag for implicit actuators
        # if this is false, we by-pass certain checks when doing actuator-related operations
        self._has_implicit_actuators = False
        # iterate over all actuator configurations
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # type annotation for type checkers
            actuator_cfg: ActuatorBaseCfg
            # create actuator group
            joint_ids, joint_names = self.find_joints(actuator_cfg.joint_names_expr)
            # check if any joints are found
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            # for efficiency avoid indexing when over all indices
            if len(joint_names) == self.num_joints:
                joint_ids = slice(None)
            # create actuator collection
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=self.root_view.count,
                device=self.device,
            )
            # log information on actuator groups
            carb.log_info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}' and"
                f" joint names: {joint_names} [{joint_ids}]."
            )
            # store actuator group
            self.actuators[actuator_name] = actuator
            # set the passed gains and limits into the simulation
            if isinstance(actuator, ImplicitActuator):
                self._has_implicit_actuators = True
                # the gains and limits are set into the simulation since actuator model is implicit
                self.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(actuator.damping, joint_ids=actuator.joint_indices)
                self.write_joint_torque_limit_to_sim(actuator.effort_limit, joint_ids=actuator.joint_indices)
            else:
                # the gains and limits are processed by the actuator model
                # we set gains to zero, and torque limit to a high value in simulation to avoid any interference
                self.write_joint_stiffness_to_sim(0.0, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(0.0, joint_ids=actuator.joint_indices)
                self.write_joint_torque_limit_to_sim(1.0e9, joint_ids=actuator.joint_indices)

        # perform some sanity checks to ensure actuators are prepared correctly
        total_act_joints = sum(actuator.num_joints for actuator in self.actuators.values())
        if total_act_joints != self.num_joints:
            carb.log_warn(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                f" joints available: {total_act_joints} != {self.num_joints}."
            )

    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        # process actions per group
        for actuator in self.actuators.values():
            # prepare input for actuator model based on cached data
            # TODO : A tensor dict would be nice to do the indexing of all tensors together
            control_action = ArticulationActions(
                joint_positions=self._data.joint_pos_target[:, actuator.joint_indices],
                joint_velocities=self._data.joint_vel_target[:, actuator.joint_indices],
                joint_efforts=self._data.joint_effort_target[:, actuator.joint_indices],
                joint_indices=actuator.joint_indices,
            )
            # compute joint command from the actuator model
            control_action = actuator.compute(
                control_action,
                joint_pos=self._data.joint_pos[:, actuator.joint_indices],
                joint_vel=self._data.joint_vel[:, actuator.joint_indices],
            )
            # update targets (these are set into the simulation)
            if control_action.joint_positions is not None:
                self._joint_pos_target_sim[:, actuator.joint_indices] = control_action.joint_positions
            if control_action.joint_velocities is not None:
                self._joint_vel_target_sim[:, actuator.joint_indices] = control_action.joint_velocities
            if control_action.joint_efforts is not None:
                self._joint_effort_target_sim[:, actuator.joint_indices] = control_action.joint_efforts
            # update state of the actuator model
            if isinstance(actuator, ImplicitActuator):
                # TODO read torque from simulation (#127)
                pass
                # self._data.computed_torques[:, actuator.joint_indices] = ???
                # self._data.applied_torques[:, actuator.joint_indices] = ???
            else:
                # -- torques
                self._data.computed_torque[:, actuator.joint_indices] = actuator.computed_effort
                self._data.applied_torque[:, actuator.joint_indices] = actuator.applied_effort
                # -- actuator data
                self._data.soft_joint_vel_limits[:, actuator.joint_indices] = actuator.velocity_limit
                # TODO: find a cleaner way to handle gear ratio. Only needed for variable gear ratio actuators.
                if hasattr(actuator, "gear_ratio"):
                    self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio
