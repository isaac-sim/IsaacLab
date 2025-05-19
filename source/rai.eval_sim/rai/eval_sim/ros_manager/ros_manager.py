# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

import carb
import rclpy
import torch
from isaaclab.envs import ManagerBasedEnv
from rai.eval_sim.utils import random_actions
from rclpy.node import Node
from rclpy.qos import (
    Duration,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from .omnigraph_ros import OmniGraphTerm
from .omnigraph_ros_cfg import OmniGraphTermCfg
from .publishers import ObservationPublisherBase, PublisherBaseTerm
from .publishers_cfg import ClockPublisherCfg, PublisherBaseTermCfg
from .subscribers import SubscriberTerm
from .subscribers_cfg import SubscriberTermCfg

QOS_PROFILE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    lifespan=Duration(seconds=1000),
)


class RosManager:
    """Manager responsible for interfacing with external control inputs / output over ROS."""

    def __init__(self, cfg: object, env: ManagerBasedEnv, node: Node):
        """Initialize the ros manager.

        Args:
            cfg: The configuration object.
            env: The environment.
            node: The ros node.
        """
        self._node = node

        # store the inputs
        self.cfg = copy.deepcopy(cfg)
        self._env = env

        # Lists to store the terms so they can be accessed and closed later
        self._sync_sub_terms: dict[str, SubscriberTerm] = {}
        self._action_sub_terms: dict[str, SubscriberTerm] = {}
        self._setting_sub_terms: list[SubscriberTerm] = []
        self._pub_terms: list[PublisherBaseTerm] = list()
        self._og_terms: list[OmniGraphTerm] = list()

        # parse config to create terms information
        self._prepare_terms()

    def close(self):
        """Close the ros manager."""
        # close publisher and subscriber terms
        for term in self.active_terms:
            term.close()

        # delete all references to the environment
        del self._env

    @property
    def active_terms(self) -> list[PublisherBaseTerm | SubscriberTerm | OmniGraphTerm]:
        """Returns a list of all terms."""
        print(
            "Active terms:"
            f" {list(self._action_sub_terms.values()) + self._setting_sub_terms + self._pub_terms + self._og_terms}"
        )
        return list(self._action_sub_terms.values()) + self._setting_sub_terms + self._pub_terms + self._og_terms

    def subscribe(self, num_envs: int = 1) -> torch.Tensor | None:
        """Checks for any new messages from the subscribers and returns any action messages received.

        Args:
            num_envs: The number of environments. Defaults to 1.

        Returns:
            If there are no action subscribers configured, returns a random action tensor.
            If there are action subscribers configured, returns:
                New actions as a tensor if any new action messages have arrived,
                None if no new action messages have arrived.
        """
        print(f"Subscribing to actions for {num_envs} environments")
        if len(self._action_sub_terms) == 0:
            carb.log_warn("No action subscribers found. Returning random actions.")
            return random_actions(self._env)

        actions = dict()
        sync_received = []
        while True:
            # Spin ros node to get any new messages
            rclpy.spin_once(self._node, timeout_sec=self.cfg.lockstep_timeout)
            # Get any actions from the subscribers
            for term_name, action_sub_term in self._action_sub_terms.items():
                # Check subscribers that haven't been synced this step
                # Check that the msg exists for this step
                # async messages will always be checked
                if action_sub_term.msg_converted is not None and term_name not in sync_received:
                    # add converted message to action buffer
                    actions[term_name] = action_sub_term.msg_converted
                    # Set converted to None so we don't reuse the same message next cycle for sync messages
                    # async messages will not be over written so they can be added in next loop
                    if action_sub_term.cfg.sync_sub:
                        sync_received.append(term_name)
                        action_sub_term.msg_converted = None

            # if no terms are used for synchronization
            if len(self._sync_sub_terms) == 0:
                break
            # check that all synchronizing messages were received
            if len(self._sync_sub_terms) == len(sync_received):
                break

        if len(actions) > 0:
            if len(actions) != len(self._action_sub_terms):
                raise RuntimeError(
                    f"Received {len(actions)} actions, but expected {len(self._action_sub_terms)}. "
                    "Double check that all action subscribers are publishing at the same rate."
                )
            # We have new action(s) to apply to the environment
            # Convert to tensor, expand for the case of multiple envs, and move to device
            return torch.concat(list(actions.values())).expand(num_envs, -1).to(self._env.device)
        else:
            # No new actions were received
            return None

    def publish_static(self, obs: dict | None = None):
        """Publish static observations through ROS, (i.e. publishers with substep=None)"""
        for term in self._pub_terms:
            if term.cfg.substep is None:
                term.publish(obs)

    def publish(self, obs: dict):
        """Publish observations through ROS."""
        for term in self._pub_terms:
            if term.cfg.substep is not None:
                if term.cfg.substep < 1:
                    raise ValueError(
                        f"Continuous publisher substep must be greater than 1, received: {term.cfg.substep}"
                    )

                if self._env.sim.current_time_step_index % term.cfg.substep == 0:
                    # publish each substep
                    if isinstance(term, ObservationPublisherBase):
                        term.publish(obs)
                    else:
                        term.publish()

    def _prepare_terms(self):
        """Initializes each ROS term."""

        # Go through all the terms in the config and create them
        for term_name, term_cfg in self.cfg.__dict__.items():
            if isinstance(term_cfg, SubscriberTermCfg):
                try:
                    term = term_cfg.class_type(term_cfg, self._node, self._env, QOS_PROFILE)
                except Exception:
                    err_msg = f"SubscriberTermCfg: '{term_name}' failed to load"
                    carb.log_error(err_msg)
                    raise Exception(err_msg)
                else:
                    if term.subscriber_type == "action":
                        self._action_sub_terms[term_name] = term
                        if term_cfg.sync_sub:
                            self._sync_sub_terms[term_name] = term
                    elif term.subscriber_type == "setting":
                        self._setting_sub_terms.append(term)

            elif isinstance(term_cfg, PublisherBaseTermCfg):
                try:
                    term = term_cfg.class_type(
                        term_cfg,
                        self._node,
                        self._env,
                        QOS_PROFILE,
                        self.cfg.use_sim_time,
                    )
                except Exception:
                    err_msg = f"PublisherBaseTermCfg: '{term_name}' failed to load"
                    carb.log_error(err_msg)
                    raise Exception(err_msg)
                else:
                    self._pub_terms.append(term)

            elif isinstance(term_cfg, OmniGraphTermCfg):
                try:
                    term = term_cfg.class_type(term_cfg, self._env)
                except Exception:
                    err_msg = f"OmniGraphTermCfg: '{term_name}' failed to load"
                    carb.log_error(err_msg)
                    raise Exception(err_msg)
                else:
                    self._og_terms.append(term)

        # ClockPublisher is a special case that is always added
        try:
            term_cfg = ClockPublisherCfg()
            term = term_cfg.class_type(
                term_cfg,
                self._node,
                self._env,
                QOS_PROFILE,
                self.cfg.use_sim_time,
            )
        except Exception as e:
            err_msg = f"ClockPublisher was unable to be created: {e}"
            carb.log_error(err_msg)
            raise Exception("ClockPublisher was unable to be created") from e
        else:
            self._pub_terms.append(term)
