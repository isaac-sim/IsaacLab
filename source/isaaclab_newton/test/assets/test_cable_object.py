# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import warp as wp

pytest.importorskip("newton")

from isaaclab_newton.assets import CableObject as NewtonCableObject
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

from isaaclab.assets import CableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable import VBDSolverCfg


@configclass
class _CableSceneCfg(InteractiveSceneCfg):
    cable = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=CableCfg(
            positions=((0.0, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.4, 1.0), (0.0, 0.6, 1.0)),
            physics_material=CableMaterialCfg(
                thickness=0.02,
                density=500.0,
                stretch_stiffness=1.0e5,
                bend_stiffness=1.0e3,
            ),
        ),
    )


def _expected_segment_state(cable, state, model) -> tuple[torch.Tensor, torch.Tensor]:
    root_body_ids = wp.to_torch(cable.root_view.get_attribute("joint_parent", model)[:, 0, 0]).long()
    root_pose = wp.to_torch(state.body_q)[root_body_ids].unsqueeze(1)
    root_velocity = wp.to_torch(state.body_qd)[root_body_ids].unsqueeze(1)
    link_pose = wp.to_torch(cable.root_view.get_link_transforms(state)[:, 0])
    link_velocity = wp.to_torch(cable.root_view.get_link_velocities(state)[:, 0])
    return torch.cat((root_pose, link_pose), dim=1), torch.cat((root_velocity, link_velocity), dim=1)


def test_interactive_scene_manages_newton_cables():
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cpu",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(_CableSceneCfg(num_envs=2, env_spacing=1.0))
        sim.reset()
        scene.update(0.0)

        cable = scene["cable"]
        assert isinstance(cable, NewtonCableObject)
        assert scene.cable_objects["cable"] is cable
        assert "cable" in scene.keys()
        assert cable.num_instances == 2
        assert cable.num_segments == 3
        assert cable.data.segment_pose_w.torch.shape == (2, 3, 7)
        assert cable.data.segment_velocity_w.torch.shape == (2, 3, 6)

        model = SimulationManager.get_model()
        state = SimulationManager.get_state_0()
        expected_pose, expected_velocity = _expected_segment_state(cable, state, model)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, expected_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, expected_velocity)
        assert len(model.body_color_groups) > 0

        initial_pose = cable.data.segment_pose_w.torch.clone()
        scene.reset()
        for _ in range(3):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.cfg.dt)

        assert torch.isfinite(cable.data.segment_pose_w.torch).all()
        assert torch.isfinite(cable.data.segment_velocity_w.torch).all()
        assert torch.any(cable.data.segment_pose_w.torch[..., 2] < initial_pose[..., 2])
        expected_pose, expected_velocity = _expected_segment_state(
            cable, SimulationManager.get_state_0(), SimulationManager.get_model()
        )
        torch.testing.assert_close(cable.data.segment_pose_w.torch, expected_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, expected_velocity)

        previous_state = state
        sim.reset()
        scene.update(0.0)
        state = SimulationManager.get_state_0()
        assert state is not previous_state

        expected_pose, expected_velocity = _expected_segment_state(cable, state, SimulationManager.get_model())
        torch.testing.assert_close(cable.data.segment_pose_w.torch, expected_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, expected_velocity)

        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.cfg.dt)
        assert torch.isfinite(cable.data.segment_pose_w.torch).all()
        assert torch.isfinite(cable.data.segment_velocity_w.torch).all()
