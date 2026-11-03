# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MicroDuck actions for walking, recovery and skating recipes.

Existing walking APIs retain their behavior. Descriptive task-specific variants preserve distinct
encoder storage, command sampling, contact grouping, recovery gates and non-finite handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


from isaaclab.managers import SceneEntityCfg

from .events import encoder_bias


class BiasedJointPositionAction(JointPositionAction):
    """Joint position action that compensates the joint-encoder calibration error."""

    cfg: BiasedJointPositionActionCfg
    """The configuration of the action term."""

    encoder_bias: torch.Tensor
    """Encoder calibration error per environment and action joint [rad], shape (num_envs, action_dim)."""

    def __init__(self, cfg: BiasedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.encoder_bias = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def apply_actions(self):
        target = self.processed_actions - self.encoder_bias
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the encoder-bias-compensating joint position action term."""

    class_type: type[BiasedJointPositionAction] = BiasedJointPositionAction


class ArticulationEncoderBiasAction(JointPositionAction):
    r"""Joint position action that compensates the joint-encoder calibration error.

    The servo closes its position loop on the encoder reading, so a command of :math:`q^*` parks
    the joint at :math:`q^* - b`, where :math:`b` is the encoder bias
    (:func:`~isaaclab_tasks.contrib.microduck.mdp.events.encoder_bias`). Upstream reproduces this by
    subtracting the bias from the target (reference section 3):

    .. math::

        \text{target} = \text{default} + \text{scale} \times \text{action} - \text{bias}

    This is the action-side half of the encoder-bias randomization, and it only makes sense
    together with the observation-side half,
    :func:`~isaaclab_tasks.contrib.microduck.mdp.observations.joint_pos_rel_articulation_biased`: the actor sees
    ``joint_pos + bias`` and the target subtracts ``bias``, so a policy commanding ``a`` reads back
    ``a`` while the true joint settles at ``default + a - bias``. What the bias then perturbs is
    the *robot*, not the policy's frame of reference -- the joints of a leg no longer agree on
    where zero is, which is exactly the calibration error the term models. Using only one half
    instead trains a policy against a permanent offset in its own commands.

    Both halves read the same tensor, so a task must give this term and the biased observation the
    same asset.

    .. note::
        Reference section 3 states that outcome the other way round -- true joint at ``HOME + a``,
        reading ``HOME + a + bias``. The simulator carries no bias of its own, so it tracks the
        target it is handed: the joint settles on ``default + a - bias`` and the biased observation
        reports ``q + bias = default + a``. The formulas quoted in that section are right; the one
        sentence of prose after them is not.
    """

    cfg: ArticulationEncoderBiasActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ArticulationEncoderBiasActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # resolved eagerly so a mismatched asset fails at construction rather than at the first step
        self._encoder_bias = encoder_bias(env, SceneEntityCfg(cfg.asset_name))

    @property
    def joint_ids(self) -> torch.Tensor | slice:
        """Indices of the joints this term drives, in ``asset_cfg.joint_ids`` order.

        Public accessor for the base class's ``_joint_ids``: this task family's reward terms need
        to cross-reference an action term's driven joints against their own ``asset_cfg`` selection
        (see :func:`~isaaclab_tasks.contrib.microduck.mdp.rewards.action_over_limit_penalty` and
        :class:`~isaaclab_tasks.contrib.microduck.mdp.rewards.joint_action_rate_l2`), and the base
        :class:`~isaaclab.envs.mdp.actions.joint_actions.JointAction` exposes no public equivalent.
        """
        return self._joint_ids

    def apply_actions(self):
        # written out rather than delegated: this runs once per physics step, so subtracting the
        # bias into ``processed_actions`` would compound it over the decimation window
        target = self.processed_actions - self._encoder_bias[:, self._joint_ids]
        self._asset.actuators.target_command.set_position_index(value=target, joint_ids=self._joint_ids)


@configclass
class ArticulationEncoderBiasActionCfg(JointPositionActionCfg):
    """Configuration for the encoder-bias-compensating joint position action term.

    Please refer to the :class:`ArticulationEncoderBiasAction` class for more details.
    """

    class_type: type[ArticulationEncoderBiasAction] = ArticulationEncoderBiasAction
