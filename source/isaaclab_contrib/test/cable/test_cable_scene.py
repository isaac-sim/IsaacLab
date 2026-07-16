# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_newton.physics import NewtonCfg

from isaaclab.assets import CableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable import VBDSolverCfg


@configclass
class _CableSceneCfg(InteractiveSceneCfg):
    cable: CableObjectCfg = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=CableCfg(
            positions=((0.0, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.4, 1.0)),
            physics_material=CableMaterialCfg(
                thickness=0.02,
                density=500.0,
                stretch_stiffness=1.0e5,
                bend_stiffness=1.0e3,
            ),
        ),
    )


def test_interactive_scene_manages_cloned_cable_lifecycle():
    """Test cloned cable registration, stepping, and reset through InteractiveScene."""
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

        cable = scene["cable"]
        assert scene.cables["cable"] is cable
        assert "cable" in scene.keys()
        assert cable.num_instances == 2
        assert cable.num_segments == 2

        initial_pose = cable.data.segment_pose_w.torch.clone()
        assert initial_pose.shape == (2, 2, 7)
        assert cable.data.segment_velocity_w.torch.shape == (2, 2, 6)

        scene.reset()
        for _ in range(3):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.cfg.dt)

        assert torch.isfinite(cable.data.segment_pose_w.torch).all()
        assert torch.isfinite(cable.data.segment_velocity_w.torch).all()

        stepped_pose = cable.data.segment_pose_w.torch.clone()
        stepped_velocity = cable.data.segment_velocity_w.torch.clone()
        assert torch.any(stepped_pose[..., 2] < initial_pose[..., 2])

        scene.reset(env_ids=[0])
        scene.write_data_to_sim()
        scene.update(0.0)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, stepped_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, stepped_velocity)
