# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP pieces for sysid replay environments.

Ported from the agile G1 sysid stack (``g1_sysid_base_env_cfg.py``). The env is a
minimal :class:`ManagerBasedRLEnvCfg`: no rewards, no randomization, a single
joint-position observation, and all joints driven as position targets. The
fitting script replays a recorded trajectory through it open-loop.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass


@configclass
class SysidObservationGroupCfg(ObsGroup):
    """Minimal observations — only joint positions (required for env validity)."""

    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SysidObservationsCfg:
    policy: SysidObservationGroupCfg = SysidObservationGroupCfg()


@configclass
class SysidActionsCfg:
    """All joints commanded as position targets (relative to default, scale=1.0).

    The fitting script sends ``des_dof_pos - default_joint_pos``, so with
    ``use_default_offset=True`` the applied PD target is the absolute recorded
    command.
    """

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


@configclass
class SysidEventCfg:
    """Reset only — no domain randomization (the optimizer owns the parameters)."""

    reset_robot = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class SysidRewardsCfg:
    """No rewards — sysid envs are not used for RL training."""


@configclass
class SysidTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CmaEsCfg:
    """CMA-ES meta-parameters read by ``scripts/sysid/fit.py``."""

    max_iteration: int = 200
    # Initial CMA-ES std in normalised [-1, 1] search space; 0.7 probes near the
    # bound edges from gen 0.
    sigma: float = 0.7
    epsilon: float | None = None
    save_interval: int = 10
    save_optimization_process: bool = False
    # Early-stop if the population std stays below plateau_min_delta for
    # plateau_patience iterations.
    plateau_patience: int = 10
    plateau_min_delta: float = 1e-3


@configclass
class SysIdCfg:
    """Meta-configuration read by ``scripts/sysid/fit.py`` (never by the env itself)."""

    robot_name: str = MISSING
    # Canonical joint order matching the dataset columns.
    joint_order: list[str] = MISSING
    cmaes: CmaEsCfg = CmaEsCfg()
