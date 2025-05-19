# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from diagnostic_msgs.msg import KeyValue
from isaaclab.utils import configclass
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from . import subscribers


@configclass
class SubscriberTermCfg:
    """Base config class for subscriber terms."""

    class_type: object = MISSING
    """Class associated with this SubscriberTermCfg"""
    msg_type: object = MISSING
    """ROS message type to subscribe to."""
    topic: str = "/actions"
    """ROS message topic name to subscribe to."""
    sync_sub: bool = True
    """Whether this subscriber is used to lockstep with simulation.

    Note all terms with sync_sub=True must be published at the same rate."""


@configclass
class JointCommandSubscriberCfg(SubscriberTermCfg):
    """Config class to apply position, velocity, and effort commands from a JointState message."""

    class_type = subscribers.JointCommandSubscriber
    msg_type = JointState
    topic: str = "{asset_name}_joint_command"
    joint_position_action_name: str = "joint_positions"
    """The name of the action term associated with joint positions."""
    joint_velocity_action_name: str = "joint_velocities"
    """The name of the action term associated with joint velocities."""
    joint_effort_action_name: str = "joint_efforts"
    """The name of the action term associated with joint efforts."""


@configclass
class JointPositionCommandSubscriberCfg(SubscriberTermCfg):
    """Config class to apply position commands from a JointState message."""

    class_type = subscribers.JointPositionCommandSubscriber
    msg_type = JointState
    topic: str = "{asset_name}_joint_command"
    action_name: str | None = "joint_pos"
    """The name of the JointPositionAction term to associate with subscriber.

    Defaults to 'joint_pos'.

    If None, will not associate with any action term, but will directly
    apply concatenated joint positions to the asset. This is useful when there are
    multiple action terms in the same message (e.g. joint positions and gripper
    positions).
    """


@configclass
class JointEffortCommandSubscriberCfg(SubscriberTermCfg):
    """Config class to apply effort commands from a JointState message."""

    class_type = subscribers.JointEffortCommandSubscriber
    msg_type = JointState
    topic: str = "{asset_name}_joint_command"
    action_name: str | None = "joint_effort"
    """The name of the JointEffortAction term to associate with subscriber.

    Defaults to 'joint_effort'.

    If None, will not associate with any action term, but will directly
    apply concatenated joint efforts to the asset. This is useful when there are
    multiple action terms in the same message (e.g. joint effort and gripper
    efforts).
    """


@configclass
class PDGainsSubscriberCfg(SubscriberTermCfg):
    """Config class to apply PID gains from a JointState message.

    The JointState message containing gains should have the following structure:
    - name: List of joint names
    - position: List of proportional gains
    - velocity: List of derivative gains
    - effort: Unused
    """

    class_type = subscribers.PDGainsSubscriber
    # Note this is a slight misuse of JointState, but it is the correct size and
    # avoids creating a new custom message type
    msg_type = JointState
    topic: str = "/pd_gains"
    asset_name: str = "robot"


@configclass
class ResetSubscriberCfg(SubscriberTermCfg):
    """Config class to reset the eval sim environment instance.

    See ResetSubscriber for more information on usage.
    """

    class_type = subscribers.ResetSubscriber
    msg_type = Bool
    topic: str = "/reset"


@configclass
class SimulationParametersSubscriberCfg(SubscriberTermCfg):
    """Config class to set arbitrary eval sim environment settings by name.

    See SimulationParametersSubscriber for more information on usage.
    """

    class_type = subscribers.SimulationParametersSubscriber
    msg_type = KeyValue
    topic: str = "/simulation_parameters"
