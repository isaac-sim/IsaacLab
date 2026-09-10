# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The last of mjlab's differences: its observation split and its PPO settings.

With the plant, the objective, the contact model and the terrain matched, three things were still
ours. This takes all three.

**An asymmetric actor-critic.** mjlab runs two observation groups. The actor gets the eight terms it
could plausibly have on hardware, corrupted; the critic gets the same uncorrupted plus four
privileged foot terms, and the *true* joint positions where the actor gets biased ones. We ran a
single group for both, which throws away the critic's ability to see what the actor cannot.

**Biased joint positions.** ``joint_pos_rel(biased=True)`` adds a per-episode offset to what the
actor reads -- encoder zero error, which a real G1 has and our simulation did not model at all. The
critic sees the truth, so the value function is not fooled by the same offset.

**PPO settings.** Ours against theirs, the three that differ:

===================  =========  =======
setting              mjlab      ours
===================  =========  =======
``obs_normalization``  **True**   False
``entropy_coef``       **0.01**   0.008
``max_iterations``   **30000**   6000
===================  =========  =======

The last one is not a detail. Every arm on this line takes 3000 to 4000 iterations just to leave
``success_rate`` 0.000, so a 6000-iteration budget is barely two thousand iterations of real
learning -- and mjlab gives itself **five times** that. ``mf`` and ``mc`` were both still at terrain
level 0 when their 6000 ran out; whether they were failing or merely slow is not a question 6000
iterations can answer.

Everything else was already identical: hidden sizes, activation, ``init_std``, clip, epochs,
minibatches, learning rate, adaptive schedule, gamma, lambda, ``desired_kl``, ``max_grad_norm`` and
``num_steps_per_env``.
"""

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensor
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import ObservationsCfg

from .rough_29dof_mjlab_env_cfg import _FOOT_BODIES, _LOCOMOTION_JOINTS, _sole_clearance
from .rough_29dof_mjlabterrain_env_cfg import G129DofRoughMjlabTerrainEnvCfg

_JOINT_BIAS = 0.05
"""Half-width of the per-episode joint-position offset [rad].

mjlab's ``biased=True`` flag with its default magnitude. Encoder zero error on the hardware is of
this order, and it is the one sim-to-real term that a policy cannot average away over an episode
because it does not change within one.
"""

_TERRAIN_MAX_DISTANCE = 5.0
"""``terrain_scan.max_distance``; mjlab scales the height scan by its reciprocal instead of clipping."""


def _as_tensor(value):
    """Return ``value`` as a plain tensor, unwrapping this repo's ``ProxyArray`` where present."""
    return value.torch if hasattr(value, "torch") else value


class joint_pos_rel_biased(ManagerTermBase):
    """Joint positions relative to the default pose, plus a per-episode constant offset.

    The offset is drawn once per environment at reset and held for the episode, which is what makes
    it an encoder zero error rather than noise: no amount of filtering removes it.
    """

    def __init__(self, cfg: ObsTerm, env):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._bias = torch.zeros_like(_as_tensor(asset.data.joint_pos))
        self._resample(torch.arange(env.num_envs, device=env.device))

    def _resample(self, env_ids):
        """Draw a fresh offset for the given environments."""
        shape = (len(env_ids), self._bias.shape[1])
        self._bias[env_ids] = (torch.rand(shape, device=self._bias.device) * 2.0 - 1.0) * _JOINT_BIAS

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._bias.shape[0], device=self._bias.device)
        self._resample(env_ids)

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        offset = _as_tensor(asset.data.joint_pos) - _as_tensor(asset.data.default_joint_pos) + self._bias
        return offset[:, asset_cfg.joint_ids]


def foot_clearance_obs(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=list(_FOOT_BODIES))
) -> torch.Tensor:
    """Height of each sole above the terrain beneath it, shape ``(num_envs, 2)``."""
    return _sole_clearance(env, asset_cfg)


def foot_air_time_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Time each foot has been airborne, shape ``(num_envs, 2)``."""
    contacts: ContactSensor = env.scene[sensor_cfg.name]
    return contacts.data.current_air_time[:, sensor_cfg.body_ids]


def foot_contact_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1.0 where the foot is in contact, shape ``(num_envs, 2)``."""
    contacts: ContactSensor = env.scene[sensor_cfg.name]
    return (contacts.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()


def foot_contact_forces_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact-force magnitude under each foot [N], shape ``(num_envs, 2)``."""
    contacts: ContactSensor = env.scene[sensor_cfg.name]
    return torch.norm(contacts.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)


@configclass
class MjlabObservationsCfg(ObservationsCfg):
    """Two groups: what the robot could publish, and what the critic is allowed to know."""

    @configclass
    class CriticCfg(ObservationsCfg.PolicyCfg):
        """The actor's terms uncorrupted, with true joint positions and the foot state added."""

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    critic: CriticCfg = CriticCfg()


@configclass
class G129DofRoughMjlabObsEnvCfg(G129DofRoughMjlabTerrainEnvCfg):
    """The clone with mjlab's observation split as well."""

    observations: MjlabObservationsCfg = MjlabObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        foot_contacts = SceneEntityCfg("contact_forces", body_names=list(_FOOT_BODIES))

        # The joint terms have to cover the same 29 joints as everything else, or the biased term
        # silently reports 43 and the observation grows by fourteen fingers. A *fresh* config per
        # term: the manager resolves names to ids in place, and a shared object resolved twice fails
        # with "both joint_names and joint_ids are specified".
        def robot_joints():
            return SceneEntityCfg("robot", joint_names=list(_LOCOMOTION_JOINTS))

        for group in (self.observations.policy, self.observations.critic):
            # mjlab's noise magnitudes, and its height-scan normalisation: the raw distance scaled by
            # 1 / max_distance rather than an offset and a clip.
            group.base_lin_vel.noise = Unoise(n_min=-0.5, n_max=0.5)
            group.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
            group.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
            group.joint_vel.noise = Unoise(n_min=-1.5, n_max=1.5)
            # The critic group is a fresh copy of the stock policy group, so it has not had the
            # fingers taken out of its joint terms the way the actor's have.
            group.joint_vel.params["asset_cfg"] = robot_joints()
            group.height_scan.params["offset"] = 0.0
            group.height_scan.scale = 1.0 / _TERRAIN_MAX_DISTANCE
            group.height_scan.clip = None

        self.observations.policy.joint_pos = ObsTerm(
            func=joint_pos_rel_biased,
            params={"asset_cfg": robot_joints()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": robot_joints()})
        self.observations.critic.height_scan.noise = None
        self.observations.critic.base_lin_vel.noise = None
        self.observations.critic.base_ang_vel.noise = None
        self.observations.critic.projected_gravity.noise = None
        self.observations.critic.joint_vel.noise = None

        # The asset_cfg has to be in ``params`` or the manager never resolves the body names and the
        # default in the signature matches every body on the robot.
        self.observations.critic.foot_height = ObsTerm(
            func=foot_clearance_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=list(_FOOT_BODIES))},
        )
        self.observations.critic.foot_air_time = ObsTerm(func=foot_air_time_obs, params={"sensor_cfg": foot_contacts})
        self.observations.critic.foot_contact = ObsTerm(func=foot_contact_obs, params={"sensor_cfg": foot_contacts})
        self.observations.critic.foot_contact_forces = ObsTerm(
            func=foot_contact_forces_obs, params={"sensor_cfg": foot_contacts}
        )
