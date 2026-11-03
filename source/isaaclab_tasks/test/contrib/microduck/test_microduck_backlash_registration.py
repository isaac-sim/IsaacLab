# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise CLI composition across the three distinct converted robot families."""

import pytest

import isaaclab_tasks.contrib.microduck  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("task", "asset"),
    [
        ("IsaacContrib-Velocity-Flat-MicroDuck-Pollen", "microduck_walk_backlash.usd"),
        ("IsaacContrib-StandUp-Flat-MicroDuck", "microduck_allcollisions_backlash.usd"),
        ("IsaacContrib-Velocity-Flat-MicroDuck-Rollers", "microduck_rollers_backlash.usd"),
    ],
)
def test_backlash_cli_composes_robot_encoders_and_agent(task, asset):
    standard, standard_agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    selected, selected_agent = resolve_task_config(
        task, "rsl_rl_cfg_entry_point", overrides=["presets=backlash", "env.scene.num_envs=8"]
    )
    assert selected.scene.robot.spawn.usd_path.endswith(asset)
    assert selected.scene.num_envs == 8
    assert len(selected.actions.joint_pos.joint_names) == 14
    assert selected.scene.robot.actuators["servos"].backlash_joint_template == "passive_{}_backlash"
    assert selected_agent.experiment_name == standard_agent.experiment_name + "_backlash"
    assert selected_agent.max_iterations == standard_agent.max_iterations
    assert "backlash_cfg" in selected.observations.policy.joint_pos.params
    assert "backlash_cfg" not in standard.observations.policy.joint_pos.params
    assert selected.observations.policy.joint_vel.params["min_lag"] == 1
    if task.endswith("-Rollers"):
        assert not hasattr(selected.rewards, "head_pose_tracking")
    else:
        reward = selected.rewards.head_pose_tracking
        assert "backlash_cfg" not in standard.rewards.head_pose_tracking.params
        assert reward.params["backlash_cfg"].joint_names == [
            f"passive_{name}_backlash" for name in reward.params["asset_cfg"].joint_names
        ]
        assert reward.weight == standard.rewards.head_pose_tracking.weight


@pytest.mark.parametrize("family", ["StandUp", "SitStand", "GroundPick"])
def test_missing_rough_variants_have_independent_generated_terrain(family):
    task = f"IsaacContrib-{family}-Rough-MicroDuck"
    first, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    second, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    assert first.scene.terrain.terrain_type == "generator"
    assert first.scene.terrain.terrain_generator.curriculum
    assert first.curriculum.terrain_levels is not None
    assert first.terminations.out_of_terrain_bounds.time_out
    assert first.terminations.out_of_terrain_bounds.params["distance_buffer"] == 20.3
    assert "_rough" in agent.experiment_name
    assert first.scene.terrain.terrain_generator is not second.scene.terrain.terrain_generator
    first.scene.terrain.terrain_generator.num_cols = 1
    assert second.scene.terrain.terrain_generator.num_cols == 20


def test_roller_spin_does_not_accidentally_inherit_a_new_backlash_variant():
    with pytest.raises(ValueError, match="(?i)backlash|unknown"):
        resolve_task_config(
            "IsaacContrib-Spin-Flat-MicroDuck",
            "rsl_rl_cfg_entry_point",
            overrides=["presets=backlash"],
        )


def test_rough_walking_playback_uses_bounded_noncurricular_terrain():
    cfg, _ = resolve_task_config(
        "IsaacContrib-Velocity-Rough-MicroDuck-Pollen",
        "rsl_rl_cfg_entry_point",
        play_mode=True,
        overrides=["presets=backlash"],
    )
    terrain = cfg.scene.terrain.terrain_generator
    assert (terrain.num_rows, terrain.num_cols) == (5, 5)
    assert not terrain.curriculum
    assert cfg.curriculum.terrain_levels is None
