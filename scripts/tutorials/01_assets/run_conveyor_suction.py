# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move a parcel along a belt, lift it with suction, and release it in Newton."""

import argparse
from contextlib import closing

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, SurfaceGripperCfg
from isaaclab.controllers import Conveyor, ConveyorCfg, ConveyorSurfaceCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

PARCEL_SIZE = 0.05
TOOL_OFFSET = 0.10
# The controller supplies belt traction independently of the contact material.
BELT_CONTACT = sim_utils.RigidBodyMaterialBaseCfg(static_friction=1.1e-5, dynamic_friction=1.1e-5)


@configclass
class ConveyorSceneCfg(InteractiveSceneCfg):
    """A static belt, one parcel, and a kinematic suction tool."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(size=(2.0, 1.5, 0.04), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.12)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    belt = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Belt",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.28, 0.10),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=BELT_CONTACT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.28, 0.30)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    parcel = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Parcel",
        spawn=sim_utils.CuboidCfg(
            size=(PARCEL_SIZE, PARCEL_SIZE, PARCEL_SIZE),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.05),
            physics_material=BELT_CONTACT,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.66, 0.33, 0.12)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.25, 0.0, PARCEL_SIZE / 2)),
    )
    tool = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tool",
        spawn=sim_utils.CuboidCfg(
            size=(0.035, 0.035, 0.16),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.54, 0.82)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.0, 0.23), rot=(0.0, 1.0, 0.0, 0.0)),
    )
    suction = SurfaceGripperCfg(
        prim_path="{ENV_REGEX_NS}/Tool",
        target_prim_path="{ENV_REGEX_NS}/Parcel",
        tool_offset=(0.0, 0.0, TOOL_OFFSET),
    )


def step_scene(sim: sim_utils.SimulationContext, scene: InteractiveScene, steps: int) -> None:
    """Apply staged commands and refresh the scene after each physics step."""
    dt = sim.get_physics_dt()
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def run_example(sim: sim_utils.SimulationContext) -> None:
    """Convey, pick up, and release the parcel."""
    cfg = ConveyorCfg(
        surface_specs=(ConveyorSurfaceCfg("{ENV_REGEX_NS}/Belt", velocity=0.20),),
        body_pattern=r"/Parcel$",
        body_count_per_env=1,
        startup_duration_s=0.25,
    )
    with closing(InteractiveScene(ConveyorSceneCfg(num_envs=1, env_spacing=1.0))) as scene:
        with closing(Conveyor(cfg, num_envs=scene.num_envs, sim=sim)) as conveyor:
            sim.reset()
            parcel, tool, suction = scene["parcel"], scene["tool"], scene["suction"]

            start_x = parcel.data.root_pos_w.torch[0, 0].item()
            conveyor.start()
            step_scene(sim, scene, 120)
            conveyor.stop()
            step_scene(sim, scene, 12)
            distance = parcel.data.root_pos_w.torch[0, 0].item() - start_x
            if distance <= 0.03:
                raise RuntimeError("The belt did not move the parcel.")
            print(f"Conveyor moved the parcel {distance:.2f} m.")

            # Keep the tool pointing down and place the pads just above the box.
            pose = tool.data.default_root_pose.torch.clone()
            pose[:, :3] = parcel.data.root_pos_w.torch
            pose[:, 2] += PARCEL_SIZE / 2 + TOOL_OFFSET + 0.002
            tool.write_root_pose_to_sim_index(root_pose=pose)
            close = torch.ones(scene.num_envs, device=sim.device)
            suction.set_grippers_command_index(close)
            step_scene(sim, scene, 60)
            if suction.state.numpy()[0] != 1:
                raise RuntimeError("The suction pads did not seal.")

            pickup_z = parcel.data.root_pos_w.torch[0, 2].item()
            for _ in range(120):
                pose[:, 2] += 0.0006
                tool.write_root_pose_to_sim_index(root_pose=pose)
                step_scene(sim, scene, 1)
            lift = parcel.data.root_pos_w.torch[0, 2].item() - pickup_z
            if lift <= 0.04:
                raise RuntimeError("The sealed tool did not lift the parcel.")
            print(f"Suction lifted the parcel {lift:.2f} m.")

            suction.set_grippers_command_index(-close)
            step_scene(sim, scene, 60)
            if suction.state.numpy()[0] != -1 or parcel.data.root_pos_w.torch[0, 2].item() > pickup_z + 0.01:
                raise RuntimeError("The suction tool did not release the parcel onto the belt.")
            print("Suction released the parcel onto the belt.")


def main() -> None:
    """Run the example in Newton MJWarp."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = sim_utils.SimulationCfg(
        device=args.device,
        dt=1.0 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, integrator="implicitfast", cone="elliptic"),
            use_cuda_graph=False,
        ),
    )
    with launch_simulation(cfg, args), sim_utils.build_simulation_context(sim_cfg=cfg) as sim:
        sim.set_camera_view(eye=(1.1, -0.9, 0.65), target=(0.0, 0.0, 0.06))
        run_example(sim)


if __name__ == "__main__":
    main()
