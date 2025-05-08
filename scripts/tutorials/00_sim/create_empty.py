# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# """This script demonstrates how to create a simple stage in Isaac Sim.

# .. code-block:: bash

#     # Usage
#     ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# """

# """Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import warp as wp
print("warp version:")
print(wp.__version__)
print(wp.__file__)
# """Rest everything follows."""

# from isaaclab.sim import SimulationCfg, SimulationContext


# def main():
#     """Main function."""

#     # Initialize the simulation context
#     sim_cfg = SimulationCfg(dt=0.01)
#     sim = SimulationContext(sim_cfg)
#     # Set main camera
#     sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

#     # Play the simulator
#     sim.reset()
#     # Now we are ready!
#     print("[INFO]: Setup complete...")

#     # Simulate physics
#     while simulation_app.is_running():
#         # perform step
#         sim.step()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()


# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# An example of using the environment replicator for Isaac Lab.
###########################################################################

# import numpy as np
# import warp as wp

# import newton
# import newton.core.articulation
# import newton.examples
# import newton.utils
# import newton.utils.isaaclab


# class Example:
#     def __init__(self, stage_path="example_cartpole.usd", num_envs=8):
#         self.num_envs = num_envs

#         builder, stage_info = newton.utils.isaaclab.replicate_environment(
#             newton.examples.get_asset("cartpole_prototype.usda"),
#             "/World/envs/env_0",
#             "/World/envs/env_{}",
#             num_envs,
#             (2.0, 3.0, 0.0),
#             # USD importer args
#             collapse_fixed_joints=True,
#         )

#         # from pprint import pprint
#         # pprint(builder.body_key)
#         # pprint(builder.shape_key)
#         # pprint(builder.joint_key)
#         # pprint(builder.articulation_key)
#         # pprint(stage_info)

#         up_axis = stage_info.get("up_axis") or "Z"

#         # finalize model
#         self.model = builder.finalize()
#         self.model.ground = False

#         # randomize pole angles
#         rng = np.random.default_rng()
#         pole_angles = np.pi / 16.0 - np.pi / 8.0 * rng.random(num_envs)
#         self.model.joint_q[1::2].assign(pole_angles)
#         # print(self.model.joint_q)

#         self.sim_time = 0.0
#         fps = 60
#         self.frame_dt = 1.0 / fps

#         self.sim_substeps = 10
#         self.sim_dt = self.frame_dt / self.sim_substeps

#         self.solver = newton.solvers.MuJoCoSolver(self.model)

#         self.state_0 = self.model.state()
#         self.state_1 = self.model.state()
#         self.control = self.model.control()

#         newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

#         self.renderer = None
#         if stage_path:
#             self.renderer = newton.utils.SimRendererOpenGL(
#                 path=stage_path,
#                 model=self.model,
#                 scaling=1.0,
#                 up_axis=up_axis,
#                 screen_width=1280,
#                 screen_height=720,
#                 camera_pos=(0, 3, 10),
#             )

#         self.use_cuda_graph = wp.get_device().is_cuda
#         if self.use_cuda_graph:
#             with wp.ScopedCapture() as capture:
#                 self.simulate()
#             self.graph = capture.graph

#     def simulate(self):
#         for _ in range(self.sim_substeps):
#             self.state_0.clear_forces()
#             self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
#             self.state_0, self.state_1 = self.state_1, self.state_0

#     def step(self):
#         with wp.ScopedTimer("step", active=False):
#             if self.use_cuda_graph:
#                 wp.capture_launch(self.graph)
#             else:
#                 self.simulate()
#         self.sim_time += self.frame_dt

#     def render(self):
#         if self.renderer is None:
#             return

#         with wp.ScopedTimer("render", active=False):
#             self.renderer.begin_frame(self.sim_time)
#             self.renderer.render(self.state_0)
#             self.renderer.end_frame()


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
#     parser.add_argument(
#         "--stage_path",
#         type=lambda x: None if x == "None" else str(x),
#         default="example_cartpole.usd",
#         help="Path to the output USD file.",
#     )
#     parser.add_argument("--num_frames", type=int, default=12000, help="Total number of frames.")
#     parser.add_argument("--num_envs", type=int, default=256, help="Total number of simulated environments.")

#     args = parser.parse_known_args()[0]

#     with wp.ScopedDevice(args.device):
#         example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

#         for _ in range(args.num_frames):
#             example.step()
#             example.render()

#         if example.renderer:
#             example.renderer.save()

import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import load_cfg_from_registry

env_cfg = load_cfg_from_registry("Isaac-Ant-Direct-v0", "env_cfg_entry_point")
env_cfg.scene.num_envs = 2
env_cfg.sim.use_fabric = False
env_cfg.sim.device = "cuda:0"
env = gym.make("Isaac-Ant-Direct-v0", cfg=env_cfg)
from isaaclab.sim._impl.newton_manager import NewtonManager
env.reset()

for i in range(1000):
    obs, rew, terminated, truncated, info = env.step(torch.tensor(env.action_space.sample()))
    if i % 3 == 0:
        env.render()
        NewtonManager.render()
    # breakpoint()
    # print(obs.shape)
    # print(rew.shape)
    # print(terminated.shape)
    # print(truncated.shape)
    # print(info)

