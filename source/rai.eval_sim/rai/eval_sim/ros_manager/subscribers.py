# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from abc import ABC
from typing import TYPE_CHECKING

import carb
import torch
from diagnostic_msgs.msg import KeyValue
from isaaclab.envs import ManagerBasedEnv
from rclpy.node import Node
from rclpy.qos import (
    Duration,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

if TYPE_CHECKING:
    from . import subscribers_cfg


class SubscriberTerm(ABC):
    def __init__(
        self,
        cfg: subscribers_cfg.SubscriberTermCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        callback: callable,
        subscriber_type: str = "action",
    ):
        """Initialize the ros subscriber term.

        Args:
            cfg: The configuration object.
            node: The ros node instance.
            env: The Isaac Lab environment.
            qos_profile: The quality of service profile for ROS communications.
            callable: The callback function to be called when a message is received.
            subscriber_type: The type of subscriber, either "action" or "setting".
                - "action" subscribers are used to update the action tensor that
                is sent to the environment's action_manager based on the received message.
                - "setting" subscribers are used to update the environment's settings
                based on the received message.


        Raises:
            AssertionError: If the subscriber_type is not "action" or "setting".

        """
        # store the inputs
        self.cfg = cfg
        self._env = env
        assert subscriber_type in [
            "action",
            "setting",
        ], f"Subscriber type must be either 'action' or 'setting', got {subscriber_type}"
        self.subscriber_type = subscriber_type

        self.subscriber = node.create_subscription(
            msg_type=cfg.msg_type,
            topic=cfg.topic,
            callback=callback,
            qos_profile=qos_profile,
        )

    def close(self):
        # Clear up reference to the environment
        del self._env


class JointCommandSubscriber(SubscriberTerm):
    def __init__(
        self, cfg: subscribers_cfg.JointCommandSubscriberCfg, node: Node, env: ManagerBasedEnv, qos_profile: QoSProfile
    ):
        """Initialize the joint command subscriber term.

        Args:
            cfg: The configuration object of type JointCommandSubscriberCfg.
            node: The ros node instance.
            env: The Isaac Lab environment.
            qos_profile: The quality of service profile for ROS communications.

        Raises:
            ValueError: If the joint state action terms are not in the order of position, velocity, and effort.
                This is necessary because we need to know how to concatenate the action terms.
            ValueError: If the joint names in each of the action terms are not identical.
            ValueError: If the total size of the combined action terms is not the same as the total_action_dim.

        """
        super().__init__(
            cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            callback=functools.partial(self.from_joint_state_to_torch),
            subscriber_type="action",
        )
        self.cfg: subscribers_cfg.JointCommandSubscriberCfg
        joint_state_action_terms = [
            self.cfg.joint_position_action_name,
            self.cfg.joint_velocity_action_name,
            self.cfg.joint_effort_action_name,
        ]

        # NOTE: This assumes the only action terms are position, velocity, and effort
        # which is safe for now because we are assuming concatenated action terms
        all_action_terms = self._env.action_manager.active_terms

        # Remove any action terms that are not those belonging to joint state
        joint_state_action_terms_order = [term for term in joint_state_action_terms if term in all_action_terms]

        # Verify that the order of the joint state action terms is correct
        if joint_state_action_terms_order != joint_state_action_terms:
            raise ValueError(
                "Joint state action terms should be in the order of position, velocity, and effort."
                f"Current order is {joint_state_action_terms_order}."
            )

        prev_joint_names = None
        total_action_dim = 0
        # Verify that joint names are identical and in same order in all action terms
        for joint_state_action_term in joint_state_action_terms:
            action_term = self._env.action_manager.get_term(joint_state_action_term)
            _, joint_names = action_term._asset.find_joints(action_term.cfg.joint_names)
            total_action_dim += len(joint_names)

            if prev_joint_names is None:
                prev_joint_names = joint_names
                self.joint_names = joint_names
                self.size = len(self.joint_names)
            else:
                if prev_joint_names != joint_names:
                    raise ValueError(
                        f"Joint names in each of the action terms: {joint_state_action_terms_order} are not identical."
                    )

        # Total size of the combined action terms should be the same as the total_action_dim
        if total_action_dim != self._env.action_manager.total_action_dim:
            raise ValueError(
                "Total size of the combined action terms should be the same as the total_action_dim."
                f"Total size of the combined action terms is {total_action_dim},"
                f"but should be {self._env.action_manager.total_action_dim}"
            )

        self.msg_converted = torch.zeros(total_action_dim)

    def from_joint_state_to_torch(self, msg: JointState):
        """Callback function for the Joint Command subscriber term that converts JointState msg position, velocity, and effort
        to torch.tensor used in actions. The order of the joints, position, velocity, and effort is preserved.

        Args:
            msg: Received JointState message to fill the action tensor.
        """
        if set(msg.name) == set(self.joint_names):
            # Reorder the joint names, position, velocity, and effort according to the joint names in the action definition
            name_to_msg_order = {name: index for index, name in enumerate(msg.name)}

            # TODO: Can we optimize this with the stacking operation below?
            position = torch.tensor([msg.position[name_to_msg_order[n]] for n in self.joint_names])
            velocity = torch.tensor([msg.velocity[name_to_msg_order[n]] for n in self.joint_names])
            effort = torch.tensor([msg.effort[name_to_msg_order[n]] for n in self.joint_names])
        else:
            carb.log_warn(
                f"Joint names in the received message {msg.name} do not match the expected joint names"
                f" {self.joint_names}. Using the ordering directly from the received message."
            )
            position = torch.tensor(msg.position)
            velocity = torch.tensor(msg.velocity)
            effort = torch.tensor(msg.effort)

        # Stack the position, velocity, and effort tensors, we can safely assume that the order is correct
        # as it is verified in the initialization
        self.msg_converted = torch.hstack([position, velocity, effort])


class JointPositionCommandSubscriber(SubscriberTerm):
    def __init__(
        self,
        cfg: subscribers_cfg.JointPositionCommandSubscriberCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
    ):
        """Initialize the Joint Position Command subscriber term.

        Args:
            cfg: The configuration object of type JointPositionCommandSubscriberCfg.
            node: The ros node instance.
            env: The Isaac Lab environment.
            qos_profile: The quality of service profile for ROS communications.
        """
        super().__init__(
            cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            callback=functools.partial(self.from_joint_position_to_torch),
            subscriber_type="action",
        )
        self.cfg: subscribers_cfg.JointPositionCommandSubscriberCfg

        # Collect joint names in order and allocate action size
        if self.cfg.action_name is not None:  # if action_name provided allocate accordingly
            action_term = self._env.action_manager.get_term(self.cfg.action_name)
            _, self.joint_names = action_term._asset.find_joints(action_term.cfg.joint_names)
            self.size = len(self.joint_names)
        else:  # else assume total action dimension
            self.size = self._env.action_manager.total_action_dim
            self.joint_names = None

        self.msg_converted = torch.zeros(self.size)

    def from_joint_position_to_torch(self, msg: JointState):
        """Callback function for the Joint Position Command subscriber term that converts JointState msg position
        to torch.tensor used in actions. The order of the joints and positions in the action definition is preserved.

        Args:
            msg: received JointState message
        """
        if self.joint_names is not None:
            if set(msg.name) == set(self.joint_names):
                # Handle the case where the joint names in the received message do not match the expected joint names
                name_to_msg_order = {name: index for index, name in enumerate(msg.name)}
                position_ordered = [msg.position[name_to_msg_order[n]] for n in self.joint_names]
            else:
                carb.log_warn(
                    f"Joint names in the received message {msg.name} do not match the expected joint names"
                    f" {self.joint_names}. Using the ordering directly from the received message."
                )
                position_ordered = msg.position

        else:
            position_ordered = msg.position

        self.msg_converted = torch.tensor(position_ordered)


class JointEffortCommandSubscriber(SubscriberTerm):
    def __init__(
        self,
        cfg: subscribers_cfg.JointEffortCommandSubscriberCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
    ):
        """Initialize the Joint Effort Command subscriber term.

        Args:
            cfg: The configuration object of type JointEffortCommandSubscriberCfg.
            node: The ros node instance.
            env: The Isaac Lab environment.
            qos_profile: The quality of service profile for ROS communications.
        """
        super().__init__(
            cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            callback=functools.partial(self.from_joint_effort_to_torch),
            subscriber_type="action",
        )
        self.cfg: subscribers_cfg.JointEffortCommandSubscriberCfg

        # Collect joint names in order and allocate action size
        if self.cfg.action_name is not None:  # if action_name provided allocate accordingly
            action_term = self._env.action_manager.get_term(self.cfg.action_name)
            _, self.joint_names = action_term._asset.find_joints(action_term.cfg.joint_names)
            self.size = len(self.joint_names)
        else:  # else assume total action dimension
            self.size = self._env.action_manager.total_action_dim
            self.joint_names = None

        self.msg_converted = torch.zeros(self.size)

    def from_joint_effort_to_torch(self, msg: JointState):
        """Member Callback function for the Joint Effort Command subscriber term that converts JointState msg effort
        to torch.tensor used in actions. The order of the joints in the action definition is preserved.

        Args:
            msg: received JointState message
        """
        if self.joint_names is not None:
            if set(msg.name) == set(self.joint_names):
                # Handle the case where the joint names in the received message do not match the expected joint names
                name_to_msg_order = {name: index for index, name in enumerate(msg.name)}
                effort_ordered = [msg.effort[name_to_msg_order[n]] for n in self.joint_names]
            else:
                carb.log_warn(
                    f"Joint names in the received message {msg.name} do not match the expected joint names"
                    f" {self.joint_names}. Using the ordering directly from the received message."
                )
                effort_ordered = msg.effort
        else:
            effort_ordered = msg.effort

        self.msg_converted = torch.tensor(effort_ordered)


class PDGainsSubscriber(SubscriberTerm):
    """Subscriber term to set P and D gains for the specified asset's actuators.

    Args:
        cfg: The configuration object of type PDGainsSubscriberCfg.
        node: The ros node instance.
        env: The Isaac Lab environment.
        qos_profile: The quality of service profile for ROS communications.
    """

    def __init__(
        self, cfg: subscribers_cfg.PDGainsSubscriberCfg, node: Node, env: ManagerBasedEnv, qos_profile: QoSProfile
    ):
        super().__init__(
            cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            callback=functools.partial(self.set_pd_gains),
            subscriber_type="setting",
        )
        self.cfg: subscribers_cfg.PDGainsSubscriberCfg

        # Store the mapping from joint name to actuator name and joint_id
        self.joint_to_actuator_order = {}
        for actuator_name, actuator in env.scene.articulations[cfg.asset_name].actuators.items():
            for joint_id, joint_name in enumerate(actuator.joint_names):
                self.joint_to_actuator_order[joint_name] = {"actuator_name": actuator_name, "joint_id": joint_id}

        self._articulation = self._env.scene.articulations[self.cfg.asset_name]

    def set_pd_gains(self, msg: JointState):
        """Set the P and D gains for the specified asset's actuators.

        Args:
            asset_name: The name of the asset to set the P and D gains for.
            msg: The message containing the P and D gains for the asset's actuators.
                It is assumed that the P and D gains are passed in the JointState
                message in the position and velocity fields, respectively. The order
                of these gains is specified by msg.name.

        """
        # Extract the P and D gains from the message
        stiffness_gains = torch.tensor(msg.position)
        damping_gains = torch.tensor(msg.velocity)

        # Construct the dictionary to store the new actuator gains
        new_actuator_gains = {}
        for actuator_name, actuator in self._articulation.actuators.items():
            new_actuator_gains[actuator_name] = {
                "stiffness": torch.zeros(actuator.num_joints),
                "damping": torch.zeros(actuator.num_joints),
            }

        # Reorder the P and D gains according to the joint names in the action definition
        for i, joint_name in enumerate(msg.name):
            # Determine which actuator the joint belongs to and the index into the actuator
            actuator_name = self.joint_to_actuator_order[joint_name]["actuator_name"]
            index_into_actuator = self.joint_to_actuator_order[joint_name]["joint_id"]

            # Set the new P and D gains for the joint in the actuator
            new_actuator_gains[actuator_name]["stiffness"][index_into_actuator] = stiffness_gains[i]
            new_actuator_gains[actuator_name]["damping"][index_into_actuator] = damping_gains[i]

        # Update actuator gains to new values
        for actuator_name, actuator in self._articulation.actuators.items():
            actuator.stiffness = new_actuator_gains[actuator_name]["stiffness"]
            actuator.damping = new_actuator_gains[actuator_name]["damping"]


class ResetSubscriber(SubscriberTerm):
    def __init__(
        self, cfg: subscribers_cfg.ResetSubscriberCfg, node: Node, env: ManagerBasedEnv, qos_profile: QoSProfile
    ):
        """Initialize the reset subscriber term.

        Example usage publishing from command-line with topic '/reset':
            .. code-block:: bash
                ros2 topic pub --once /reset std_msgs/Bool "{'data': true}"

        Args:
            cfg: The configuration object of type ResetSubscriberCfg.
            node: The ros node instance.
            env: The Isaac Lab environment.
            qos_profile: The quality of service profile for ROS communications.
        """
        super().__init__(
            cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            callback=functools.partial(self.reset_env),
            subscriber_type="setting",
        )
        self.cfg: subscribers_cfg.ResetSubscriberCfg

    def reset_env(self, msg: Bool):
        """Resets the environment.

        Args:
            msg: The message that triggers the reset, if the data field
               is set to true
        """
        if msg.data is True:
            self._env.reset()
        else:
            carb.log_warn("Reset message received with data field set to False. Ignoring the reset request.")


class SimulationParametersSubscriber(SubscriberTerm):
    def __init__(
        self,
        cfg: subscribers_cfg.SimulationParametersSubscriberCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
    ):
        """Initialize the simulation settings subscriber term.

        Example usage publishing from command-line with topic '/simulator_settings':
            .. code-block:: bash
                ros2 topic pub --once /simulation_parameters diagnostic_msgs/KeyValue "{'key': 'dt', 'value': 0.002}"

        Args:
            cfg: The configuration object of type SimulationParametersSubscriberCfg.
            node: The ros node instance.
            env: The Isaac Lab environment.
            qos_profile: The quality of service profile for ROS communications.
        """
        super().__init__(
            cfg,
            node=node,
            env=env,
            qos_profile=QoSProfile(
                history=QoSHistoryPolicy.KEEP_ALL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                lifespan=Duration(seconds=1000),
            ),
            callback=functools.partial(self.update_simulation_parameter),
            subscriber_type="setting",
        )
        self.cfg: subscribers_cfg.SimulationParametersSubscriberCfg

        self.subscriber_param_keys_and_updaters: dict[str, callable] = {
            "decimation": self.update_decimation,  # How many physics steps between updates to sim.
            "physics_dt": self.update_physics_dt,  # The time between each physics step.
        }

    def update_decimation(self, value: str) -> None:
        """Sets the `decimation` simulation parameter.

        Note: Dependent simulation parameter, `step_dt` is affected by this change.
        """

        self._env.cfg.decimation = int(value)
        carb.log_warn(f"Set sim parameter: 'decimation'={self._env.cfg.decimation}")
        carb.log_warn(f"step_dt (dt per sim/controller update) is now={self._env.step_dt}")

    def update_physics_dt(self, value: str) -> None:
        """Sets the `physics_dt` simulation parameter.

        Note: Dependent simulation parameter, step_dt is affected by this change.
        """

        physics_dt = float(value)
        self._env.cfg.sim.dt = physics_dt
        self._env.sim.set_simulation_dt(physics_dt=physics_dt)
        carb.log_warn(f"Set sim parameter: 'physics_dt'={self._env.physics_dt}")
        carb.log_warn(f"step_dt (dt per sim/controller update) is now={self._env.step_dt}")

    def update_simulation_parameter(self, msg: KeyValue) -> None:
        """Sets the environment setting.

        Args:
            msg: The message that triggered this callback.
        """
        update_fn = self.subscriber_param_keys_and_updaters.get(msg.key)

        if update_fn is None:
            carb.log_warn(
                f"Topic '{self.cfg.topic}' requests change of unknown simulation parameter '{msg.key}' to value"
                f" '{msg.value}'.  Valid parameter keys are: {list(self.subscriber_param_keys_and_updaters.keys())}."
                " Returning with no work done."
            )
            return

        update_fn(msg.value)
