# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 policy action ordering and task progress across episode boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp

from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaaclab.sim import use_stage
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray

from isaaclab_tasks.contrib.h2_sharpa import POLICY_JOINT_NAMES, phase_reward
from isaaclab_tasks.contrib.pack_agx.config.env_config import PackAgxOrinEnvCfg
from isaaclab_tasks.contrib.pick_and_place_apple.config.env_config import PnpAppleEnvCfg


@pytest.mark.parametrize("cfg_type", [PnpAppleEnvCfg, PackAgxOrinEnvCfg])
def test_policy_actions_preserve_uncontrolled_targets(cfg_type):
    """Apply ordered absolute commands without changing the reset targets of head, waist, or legs."""
    cfg = cfg_type()
    cfg.to_dict()  # Train/play serialize the complete scene before launching the simulator.
    names = sorted(cfg.scene.robot.init_state.joint_pos)
    defaults = torch.tensor([cfg.scene.robot.init_state.joint_pos[name] for name in names]).repeat(2, 1)
    targets = defaults.clone()

    def find_joints(patterns, preserve_order=False, as_proxy=False):
        indices, selected = resolve_matching_names(patterns, names, preserve_order=preserve_order)
        return (ProxyArray(wp.array(indices, dtype=wp.int32, device="cpu")) if as_proxy else indices), selected

    def set_targets(target, joint_ids):
        targets[:, joint_ids] = target

    robot = SimpleNamespace(
        cfg=cfg.scene.robot,
        num_joints=len(names),
        num_base_dofs=0,
        find_joints=find_joints,
        data=SimpleNamespace(gravity_compensation_forces=torch.arange(2 * len(names)).reshape(2, -1).float()),
        set_joint_position_target_index=set_targets,
        set_joint_effort_target_index=Mock(),
    )
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, device="cpu")
    action = cfg.actions.joint_pos.class_type(cfg.actions.joint_pos, env)
    commands = torch.linspace(-0.4, 0.4, 116).reshape(2, 58)
    action.process_actions(commands)
    action.apply_actions()
    policy_indices = [names.index(name) for name in POLICY_JOINT_NAMES]
    other_indices = [i for i, name in enumerate(names) if name not in POLICY_JOINT_NAMES]
    torch.testing.assert_close(targets[:, policy_indices], commands)
    torch.testing.assert_close(targets[:, other_indices], defaults[:, other_indices])
    assert targets[0, names.index("head_pitch_joint")] == pytest.approx(0.6)
    effort = robot.set_joint_effort_target_index.call_args.kwargs
    torch.testing.assert_close(effort["target"], robot.data.gravity_compensation_forces[:, effort["joint_ids"]])


@pytest.mark.parametrize("cfg_type", [PnpAppleEnvCfg, PackAgxOrinEnvCfg])
def test_progress_without_rewards_and_partial_reset(cfg_type):
    """Success progresses without evaluating rewards, and only the selected episode resets."""
    cfg = cfg_type().terminations.task_success
    apple = cfg_type is PnpAppleEnvCfg
    object_name, target_name = ("apple", "plate") if apple else ("agx_orin", "protective_box")
    position = torch.zeros(2, 3)
    wrist = torch.zeros(2, 2, 3)
    joints = torch.zeros(2, 4)
    robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=wrist, joint_pos=joints),
        find_bodies=lambda names, **kwargs: ([0, 1], names) if isinstance(names, list) else ([1], [names]),
        find_joints=lambda names, **kwargs: (list(range(4)), names) if isinstance(names, list) else ([0], [names]),
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=ProxyArray(wp.from_torch(position, dtype=wp.vec3f)),
            root_quat_w=ProxyArray(wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2), dtype=wp.quatf)),
        )
    )
    target = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3)))
    env = SimpleNamespace(
        scene={object_name: obj, target_name: target, "robot": robot}, num_envs=2, device="cpu", step_dt=0.1
    )
    term = cfg.func(cfg, env)
    env.termination_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term))
    term.reset()
    hold_steps = cfg.params["hold_steps" if apple else "lift_hold_steps"]
    for phase in range(4):
        position[:, 2] = (0.2 if phase < 2 else 0.05) if apple else (0.1 if phase < 2 else 0.0)
        wrist[:, 0, 0] = 0.0 if phase == 0 else 1.0
        wrist[:, 1, 0] = 0.0 if phase < 3 else 0.5
        for _ in range(hold_steps):
            done = term(env, **cfg.params)
        assert done.tolist() == [phase == 3, phase == 3]
        torch.testing.assert_close(phase_reward(env, phase), torch.full((2,), 10.0))
        assert not phase_reward(env, (phase + 1) % 4).any()
    # Remaining in the success state must not grant another transition reward.
    term(env, **cfg.params)
    assert not phase_reward(env, 3).any()
    position[0, 2] = 2.0
    term.reset(torch.tensor([0]))
    assert term(env, **cfg.params).tolist() == [False, True]


def test_prop_material_overrides_clone_with_the_prototype(tmp_path):
    """Clones receive the shader overrides and keep the authored diffuse texture."""
    path = str(tmp_path / "prop.usda")
    source = Usd.Stage.CreateNew(path)
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/Prop").GetPrim())
    shader = UsdShade.Shader.Define(source, "/Prop/Shader")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
    source.GetRootLayer().Save()
    cfg = PackAgxOrinEnvCfg().scene.agx_orin.spawn.replace(usd_path=path)
    stage = Usd.Stage.CreateInMemory()
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    with use_stage(stage):
        cfg.func("/World/env_.*/Prop", cfg)
    for index in range(2):
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"/World/env_{index}/Prop/Shader"))
        assert shader.GetInput("diffuse_texture").Get().path == "texture.png"
        assert shader.GetInput("metallic_constant").Get() == pytest.approx(cfg.metallic)
        assert shader.GetInput("reflection_roughness_constant").Get() == pytest.approx(cfg.roughness)
        assert shader.GetInput("albedo_brightness").Get() == pytest.approx(cfg.brightness)
