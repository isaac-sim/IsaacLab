# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
The script uses a custom H1 environment configuration that includes:
- A 5x5m grid pattern ray caster attached to the H1 robot
- Multiple obstacle objects spawned in each environment for ray casting demonstration

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/isaaclab/test/sensors/check_multi_mesh_ray_caster_h1_locomotion.py --num_objects 10

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates a test with the H1 rough terrain environment with ray caster."
)
parser.add_argument("--num_objects", type=int, default=10, help="Number of obstacle objects to spawn per environment.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random

import cv2
import torch
from rsl_rl.runners import OnPolicyRunner

from isaacsim.sensors.camera import Camera

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

TASK = "Isaac-Velocity-Rough-H1-v0"
RL_LIBRARY = "rsl_rl"


@configclass
class H1RoughEnvCfg_PLAY_WITH_RAYCASTER(H1RoughEnvCfg_PLAY):
    """H1 rough environment configuration for interactive play with obstacles and multi-mesh ray caster."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Add obstacles individually to the scene
        # Default number of obstacles
        num_obstacles = 10

        for i in range(num_obstacles):
            # Add each obstacle as an individual attribute to the scene
            setattr(
                self.scene,
                f"obstacle_{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/obstacle_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.1 + random.random() * 0.5, 0.5 + random.random() * 0.5, 0.5 + random.random() * 0.05),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.0 + i / num_obstacles, 1.0 - i / num_obstacles)
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg((0.0 + random.random(), 0.0 + random.random(), 1.0)),
                ),
            )

        # Add multi-mesh ray caster with 5x5m grid pattern
        self.scene.multi_mesh_ray_caster = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/envs/env_.*/obstacle_.*", track_mesh_transforms=True
                ),
            ],
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.1,  # 10cm resolution
                size=(5.0, 5.0),  # 5x5 meter grid
            ),
            attach_yaw_only=True,
            debug_vis=True,
            update_period=0.02,  # Update at 50Hz
        )

        # Add events to reset obstacles
        for i in range(num_obstacles):
            setattr(
                self.events,
                f"reset_obstacle_{i}",
                EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": (-0.5, 0.5),
                            "y": (-0.5, 0.5),
                            "z": (0.0, 0.0),
                            "roll": (-3.14, 3.14),
                            "pitch": (-3.14, 3.14),
                            "yaw": (-3.14, 3.14),
                        },
                        "velocity_range": {
                            "x": (-0.0, 0.0),
                            "y": (-0.0, 0.0),
                            "z": (-0.0, 0.0),
                        },
                        "asset_cfg": SceneEntityCfg(f"obstacle_{i}"),
                    },
                ),
            )

        # change the visualizer to follow env 1
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        # turn off the velocity marker
        self.commands.base_velocity.debug_vis = False


class H1RoughDemoWithRayCaster:
    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        # load the trained jit policy
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)

        # Create environment with ray caster and obstacles
        env_cfg = H1RoughEnvCfg_PLAY_WITH_RAYCASTER()

        # Configure scene for interactive demo
        env_cfg.scene.num_envs = 10
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)

        # Update ray caster debug visualization based on headless mode
        if hasattr(env_cfg.scene, "multi_mesh_ray_caster"):
            env_cfg.scene.multi_mesh_ray_caster.debug_vis = not args_cli.headless

        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        self.num_envs = env_cfg.scene.num_envs

        # Access the ray caster from the scene
        if "multi_mesh_ray_caster" in self.env.unwrapped.scene.sensors:
            self.ray_caster = self.env.unwrapped.scene.sensors["multi_mesh_ray_caster"]
            print(f"Ray caster loaded from scene with {self.ray_caster.num_rays} rays per robot")
        else:
            self.ray_caster = None
            print("Warning: Ray caster not found in scene")

        # load previously trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")


def main():
    """Main function."""
    demo_h1 = H1RoughDemoWithRayCaster()
    obs, _ = demo_h1.env.reset()

    camera = Camera(prim_path="/World/floating_camera", resolution=(3600, 2430))
    camera.set_world_pose(
        position=demo_h1.env.unwrapped.scene["robot"].data.root_pos_w[0, :].cpu().numpy() + [-20, 0.0, 15],
        orientation=[0.9396926, 0.0, 0.3420201, 0.0],
    )
    camera.initialize()
    os.makedirs("images", exist_ok=True)

    # Simulation step counter for periodic updates
    step_count = 0

    while simulation_app.is_running():
        # check for selected robots

        with torch.inference_mode():
            action = demo_h1.policy(obs)
            obs, _, _, _ = demo_h1.env.step(action)
            # overwrite command based on keyboard input
            obs[:, 9:13] = demo_h1.commands

            if step_count > 10:
                camera.get_current_frame()
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(camera.get_rgba()[:, :, :3], cv2.COLOR_RGB2BGR)
                # Save the image as PNG
                assert cv2.imwrite(f"images/img_{str(step_count - 10).zfill(4)}.png", image_bgr)

        step_count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
