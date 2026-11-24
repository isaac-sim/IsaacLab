# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The calibrated asset pack is opt-in and preserves the legacy task contract."""

import os
from types import SimpleNamespace

import torch


def test_calibrated_scene_does_not_change_legacy_defaults():
    from isaaclab_tasks.contrib.assemble_trocar.calibrated_env_cfg import G1AssembleTrocarCalibratedEnvCfg
    from isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg import G1AssembleTrocarEnvCfg

    calibrated = G1AssembleTrocarCalibratedEnvCfg()
    legacy = G1AssembleTrocarEnvCfg()
    assert legacy.scene.front_camera.width == legacy.scene.front_camera.height == 224
    assert legacy.scene.robot.spawn.usd_path.startswith("https://")
    assert legacy.scene.robot.init_state.joint_pos["left_shoulder_pitch_joint"] == -0.754599
    assert (legacy.decimation, legacy.sim.dt, legacy.episode_length_s) == (4, 1 / 200, 20.0)
    assert legacy.rewards.update_stage.params["lift_threshold"] == 0.15
    assert "use_com_for_lift" not in legacy.rewards.update_stage.params
    assert (calibrated.scene.front_camera.width, calibrated.scene.front_camera.height) == (320, 240)
    asset_root = os.environ.get("ASSET_LOCAL", "/workspace/assets").rstrip("/")
    assert calibrated.scene.robot.spawn.usd_path == f"{asset_root}/Robots/g1_29dof_with_dex3_base_fix02.usd"
    assert calibrated.scene.robot.init_state.joint_pos["left_shoulder_pitch_joint"] == -0.699552
    assert (calibrated.decimation, calibrated.sim.dt, calibrated.episode_length_s) == (8, 1 / 240, 25.0)
    assert calibrated.rewards.update_stage.params["lift_threshold"] == 0.08
    assert calibrated.rewards.update_stage.params["use_com_for_lift"] is True


def test_offset_instrument_uses_com_only_when_requested():
    from isaaclab_tasks.contrib.assemble_trocar.mdp.rewards import lift_trocars_reward

    def position(z):
        return SimpleNamespace(torch=torch.tensor([[0.0, 0.0, z]]))

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=1 / 30,
        scene={
            "trocar_1": SimpleNamespace(data=SimpleNamespace(root_pos_w=position(0.90), root_com_pos_w=position(0.95))),
            "trocar_2": SimpleNamespace(data=SimpleNamespace(root_pos_w=position(0.96))),
        },
    )
    options = dict(table_height=0.85, lift_threshold=0.08, use_sparse_reward=False)
    assert lift_trocars_reward(env, **options).item() == 0
    assert lift_trocars_reward(env, **options, use_com_for_lift=True).item() == 1
