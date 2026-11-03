# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test assembled VelStand configuration and the native production-preset environment.

Numeric expectations are independent constants; focused tests cover reset and teacher behavior.
"""

import math
import os
from pathlib import Path

os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym
import pytest
import torch

from isaaclab.app import launch_simulation
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import to_dict

from isaaclab_tasks.contrib.microduck.terrain import FOOT_SITE_PATHS
from isaaclab_tasks.utils import resolve_task_config


def resolved(terrain="Rough", backlash=True):
    return resolve_task_config(
        f"IsaacContrib-VelStand-{terrain}-MicroDuck",
        "rsl_rl_cfg_entry_point",
        overrides=["presets=backlash"] if backlash else [],
    )


@pytest.mark.unit
@pytest.mark.parametrize(("terrain", "backlash"), [("Flat", False), ("Flat", True), ("Rough", False), ("Rough", True)])
def test_cli_presets_select_independent_assets_logs_and_serialized_terrain_rays(terrain, backlash):
    env, agent = resolved(terrain, backlash)
    suffix = "_backlash" if backlash else ""
    assert Path(env.scene.robot.spawn.usd_path).name == f"microduck_fullcontact{suffix}.usd"
    assert agent.experiment_name == f"microduck_velstand_{terrain.lower()}{suffix}"
    assert tuple(env.selected_presets) == (("backlash",) if backlash else ())
    assert env.scene.terrain.terrain_type == ("plane" if terrain == "Flat" else "generator")
    snapshot = to_dict(env)
    assert tuple(snapshot["selected_presets"]) == tuple(env.selected_presets)
    if terrain == "Rough":
        for name, site in zip(("left_foot_height", "right_foot_height"), FOOT_SITE_PATHS, strict=True):
            sensor = snapshot["scene"][name]
            assert sensor["prim_path"] == site
            assert sensor["global_world_only"]
            assert sensor["pattern_cfg"]["resolution"] == 0.04
            assert tuple(sensor["pattern_cfg"]["size"]) == (0.08, 0.0)
        assert env.rewards.foot_clearance.params["target_height"] == 0.02
    else:
        assert "left_foot_height" not in snapshot["scene"]
        assert env.rewards.foot_clearance.params["target_height"] == pytest.approx(0.0425)


@pytest.mark.unit
def test_merged_warm_start_curricula_and_protection_weights_match_source():
    cfg, agent = resolved()
    assert agent.max_iterations == 6000 and agent.num_steps_per_env == 24 and agent.save_interval == 250
    assert cfg.rewards.action_rate_l2.weight == -1.0
    assert cfg.rewards.head_pose_bias.weight == 3.0
    for name in (
        "action_rate_weight",
        "head_pose_bias_weight",
        "standing_envs",
        "head_pose_range",
        "com_range",
        "head_com_range",
    ):
        term = getattr(cfg.curriculum, name)
        stages = next(value for key, value in term.params.items() if key.endswith("_stages"))
        assert len(stages) == 1 and stages[0]["step"] == 0
    assert [s["step"] for s in cfg.curriculum.ground_state_mix.params["param_stages"]] == [0, 3600, 16800, 24000, 33600]
    assert [s["params"]["prone_prob"] for s in cfg.curriculum.ground_state_mix.params["param_stages"]] == [
        0.0,
        0.0,
        0.15,
        0.3,
        0.45,
    ]
    assert [s["params"]["crouch_prob"] for s in cfg.curriculum.ground_state_mix.params["param_stages"]] == [
        0.0,
        0.2,
        0.2,
        0.2,
        0.2,
    ]
    assert cfg.curriculum.fell_over_disable.params["param_stages"][-1] == {
        "step": 2400,
        "params": {"limit_angle": math.pi},
    }
    for name, final in [("fallen_tax", -0.5), ("recovery_success", 10.0), ("com_upward_velocity", 2.0)]:
        assert getattr(cfg.curriculum, f"{name}_weight").params["weight_stages"] == [
            {"step": 0, "weight": 0.0},
            {"step": 14400, "weight": final},
        ]
    for name, final in [
        ("servo_impact", -0.02),
        ("head_impact", -0.01),
        ("trunk_impact", -0.01),
        ("servo_acc_spike", -0.001),
        ("servo_stall", -0.05),
        ("gentle_rise", 0.005),
    ]:
        assert getattr(cfg.rewards, name).weight == final * 0.25
        assert getattr(cfg.curriculum, f"{name}_weight").params["weight_stages"][-1] == {"step": 24000, "weight": final}
    assert cfg.events.topple_push.interval_range_s == (4.0, 8.0)
    assert cfg.curriculum.topple_push_range.params["inclusive"] is False
    assert cfg.curriculum.topple_push_range.params["param_stages"][-1] == {
        "step": 19200,
        "params": {"velocity_range": {"x": (-1.2, 1.2), "y": (-1.2, 1.2)}},
    }
    assert cfg.commands.base_velocity.rel_turn_in_place_envs == 0.2
    assert cfg.commands.base_velocity.turn_in_place_min_frac == 0.2
    assert cfg.rewards.track_yaw_fine.weight == 2.0 and cfg.rewards.track_yaw_fine.params["std"] == 0.35
    assert cfg.rewards.body_pose_tracking.weight == 2.0
    assert cfg.commands.body_pose.zero_command_prob == 0.3
    assert cfg.commands.body_pose.axis_zero_prob[2] == 0.5
    assert cfg.curriculum.body_pose_range.params["range_stages"][-1]["step"] == 7200


@pytest.mark.unit
def test_observation_order_and_control_contract_survive_preset_composition():
    cfg, agent = resolved()

    def terms(group):
        return [name for name, value in vars(group).items() if isinstance(value, ObservationTermCfg)]

    assert terms(cfg.observations.policy) == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "velocity_commands",
        "head_pose_commands",
        "body_pose_commands",
    ]
    assert terms(cfg.observations.critic) == [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "velocity_commands",
        "foot_height",
        "foot_air_time",
        "foot_contact",
        "foot_contact_forces",
        "head_pose_commands",
        "body_pose_commands",
    ]
    assert len(cfg.actions.joint_pos.joint_names) == 14 and cfg.actions.joint_pos.scale == 1.0
    assert cfg.actions.joint_pos.clip is None
    assert cfg.sim.dt == 0.005 and cfg.decimation == 4 and cfg.episode_length_s == 20.0
    assert agent.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert agent.algorithm.bc_cfg["unanchor_turn_in_place"] is False
    assert agent.algorithm.bc_cfg["expert_mask_cols"] == tuple(range(6, 48))
    assert agent.algorithm.bc_cfg["body_slice"] == (55, 61)


@pytest.mark.integration
@pytest.mark.usefixtures("microduck_fullcontact_backlash_usd")
def test_native_production_preset_resets_and_steps_with_live_protection_channels():
    cfg, _ = resolved()
    cfg.scene.num_envs = 8
    cfg.seed = 42
    cfg.sim.device = "cuda:0"
    cfg.sim.visualizer_cfgs = []
    cfg.events.set_ground_state.params.update(sit_prob=0.0, so3_prob=0.0, joint_random_prob=0.0)
    cfg.curriculum.ground_state_mix.params["param_stages"] = [
        {"step": 0, "params": {"prone_prob": 1.0, "crouch_prob": 0.0}}
    ]
    cfg.curriculum.fell_over_disable.params["param_stages"] = [{"step": 0, "params": {"limit_angle": math.pi}}]
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-VelStand-Rough-MicroDuck", cfg=cfg)
        try:
            obs, _ = env.reset()
            assert obs["policy"].shape == (8, 61) and obs["critic"].shape == (8, 76)
            robot = env.unwrapped.scene["robot"]
            assert len(robot.find_joints(["passive_.*_backlash"])[0]) == 14
            max_effort, max_force = 0.0, 0.0
            for _ in range(32):
                obs, reward, *_ = env.step(torch.zeros(8, 14, device=cfg.sim.device))
                assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
                max_effort = max(max_effort, robot.actuators.applied_effort.torch.abs().max().item())
                for name in ("servo_ground_contact", "head_ground_contact", "trunk_ground_contact"):
                    force = env.unwrapped.scene[name].data.force_matrix_w.torch
                    assert torch.isfinite(force).all()
                    max_force = max(max_force, force.norm(dim=-1).max().item())
            assert max_effort > 0.001
            assert max_force > 0.0
        finally:
            env.close()
