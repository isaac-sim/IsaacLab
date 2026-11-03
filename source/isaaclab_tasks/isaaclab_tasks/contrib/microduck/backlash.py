# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose the gearbox-play domain preset without changing the standard task."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import clone

from isaaclab_tasks.utils import PresetCfg, preset

from .mdp.observations import delayed_observation
from .mdp.observations import joint_pos_rel_articulation_biased as joint_pos_rel_biased

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
    from isaaclab.managers import ObservationTermCfg

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def joint_pos_rel_backlash_biased(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    backlash_cfg: SceneEntityCfg,
    biased: bool = True,
) -> torch.Tensor:
    """Read servo plus paired gearbox angle, relative to HOME, with servo-only encoder bias."""
    position = joint_pos_rel_biased(env, asset_cfg, biased=biased)
    return position + env.scene[backlash_cfg.name].data.joint_pos.torch[:, backlash_cfg.joint_ids]


def joint_vel_rel_backlash(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, backlash_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Read output-side encoder velocity in corresponding servo/play order."""
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    velocity = velocity + env.scene[backlash_cfg.name].data.joint_vel.torch[:, backlash_cfg.joint_ids]
    return velocity - asset.data.default_joint_vel.torch[:, asset_cfg.joint_ids]


def _default(value: Any) -> Any:
    return value.default if isinstance(value, PresetCfg) else value


def _encoder_preset(term: ObservationTermCfg, func: Callable[..., torch.Tensor], joint_names: list[str]) -> PresetCfg:
    term = _default(term)
    variant = clone(term)
    play = SceneEntityCfg(
        "robot", joint_names=[f"passive_{name}_backlash" for name in joint_names], preserve_order=True
    )
    if variant.func is delayed_observation:
        variant.params["term_func"] = func
        variant.params["term_params"] = {**variant.params["term_params"], "backlash_cfg": play}
    else:
        variant.func = func
        variant.params["backlash_cfg"] = play
    return preset(default=term, backlash=variant)


def apply_backlash_preset(env_cfg: ManagerBasedRLEnvCfg, backlash_robot_cfg: ArticulationCfg) -> ManagerBasedRLEnvCfg:
    """Add the upstream gearbox-play alternative after constructing the task config.

    The supplied robot variant owns the converted asset and native BAM feedback.
    The task's initial pose and prim path are retained. Paired encoder observations
    keep their noise and latency. Head rewards measure servo plus paired play,
    matching upstream's shared helpers. Pose and limit joint selections exclude
    the additional passive play joints.

    Existing robot and observation preset wrappers are accepted through their
    standard alternatives, allowing callers to compose this helper once at the
    end of an environment factory. This mutates and returns the supplied config.
    """
    original = _default(env_cfg.scene.robot)
    robot = clone(backlash_robot_cfg)
    robot.prim_path = original.prim_path
    robot.init_state = clone(original.init_state)
    play_names = [f"passive_{name}_backlash" for name in env_cfg.actions.joint_pos.joint_names]
    covered = {
        name for name in play_names if any(re.fullmatch(pattern, name) for pattern in robot.init_state.joint_pos)
    }
    if not covered:
        robot.init_state.joint_pos["passive_.*_backlash"] = 0.0
    else:
        for name in play_names:
            if name not in covered:
                robot.init_state.joint_pos[name] = 0.0
    env_cfg.scene.robot = preset(default=original, backlash=robot)
    joint_names = list(env_cfg.actions.joint_pos.joint_names)
    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name)
        group.joint_pos = _encoder_preset(group.joint_pos, joint_pos_rel_backlash_biased, joint_names)
        group.joint_vel = _encoder_preset(group.joint_vel, joint_vel_rel_backlash, joint_names)

    for name in ("head_pose_tracking", "head_pose_bias"):
        term = getattr(env_cfg.rewards, name, None)
        if term is None:
            continue
        term = _default(term)
        variant = clone(term)
        selection = variant.params["asset_cfg"]
        variant.params["backlash_cfg"] = SceneEntityCfg(
            selection.name,
            joint_names=[f"passive_{name}_backlash" for name in selection.joint_names],
            preserve_order=True,
        )
        setattr(env_cfg.rewards, name, preset(default=term, backlash=variant))

    limits = getattr(env_cfg.rewards, "dof_pos_limits", None)
    if limits is not None:
        limits = _default(limits)
        variant = clone(limits)
        variant.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)
        env_cfg.rewards.dof_pos_limits = preset(default=limits, backlash=variant)

    pose = getattr(env_cfg.rewards, "pose", None)
    if pose is not None:
        pose = _default(pose)
        if "asset_cfg" in pose.params:
            variant = clone(pose)
            selection = variant.params["asset_cfg"]
            names = selection.joint_names
            if isinstance(names, str):
                names = [names]
            selection.joint_names = [r"^(?!passive_.*_backlash)" + name.lstrip("^") for name in names]
            env_cfg.rewards.pose = preset(default=pose, backlash=variant)

    # Every passive hinge can add a limit/friction row. Keep capacity changes
    # local to the domain preset, including when physics has its own variants.
    physics = env_cfg.sim.physics
    options = (
        (getattr(physics, field.name) for field in fields(physics)) if isinstance(physics, PresetCfg) else (physics,)
    )
    seen = set()
    for option in options:
        solver = getattr(option, "solver_cfg", None)
        if solver is None or id(solver) in seen or not hasattr(solver, "njmax"):
            continue
        seen.add(id(solver))
        capacity = _default(solver.njmax)
        solver.njmax = preset(default=capacity, backlash=max(128, capacity + 28))
    return env_cfg


def backlash_agent_preset(agent_cfg: RslRlOnPolicyRunnerCfg) -> RslRlOnPolicyRunnerCfg:
    """Keep standard and gearbox-play checkpoints in separate experiment directories."""
    name = _default(agent_cfg.experiment_name)
    agent_cfg.experiment_name = preset(default=name, backlash=f"{name}_backlash")
    return agent_cfg
