# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark non-RL environment."""

"""Launch Isaac Sim Simulator first."""


from isaaclab.app import AppLauncher

args = {
    "headless": True,
    "livestream": 0,
    "enable_cameras": False,
    "xr": False,
    "device": "cuda:0",
    "cpu": False,
    "verbose": False,
    "info": False,
    "experience": "",
    "rendering_mode": "performance",
    "kit_args": "",
    "anim_recording_enabled": False,
    "anim_recording_start_time": 0,
    "anim_recording_stop_time": 10,
    "num_envs": 4096,
    "task": "IsaacLab_Isaac-Cartpole-Direct-v0",
    "seed": 1,
    "num_frames": 10000,
}

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

# enable benchmarking extension
import gymnasium as gym
import os
import torch

from isaaclab.utils.timer import Timer
from isaaclab.utils.timer._timer import _class_group_registry, toggle_timer_group

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


class SuiteNonRLBenchmark:
    def __init__(self, task: str, num_envs: int, seed: int, num_frames: int, args: dict):
        self.task = task
        self.num_envs = num_envs
        self.seed = seed
        self.num_frames = num_frames
        self.args = args

        self.refined_timings = {}

    def enable_timer_groups(self):
        """Enable all timer groups."""
        for group in _class_group_registry.keys():
            toggle_timer_group(group, True)

    def instantiate_env_cfg(self):
        env_cfg = load_cfg_from_registry(self.task, "env_cfg_entry_point")
        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = self.num_envs
        env_cfg.sim.device = self.args["device"]
        env_cfg.seed = self.seed
        env = gym.make(self.task, cfg=env_cfg)
        env.reset()
        return env_cfg, env

    def run(self):
        """Run the benchmark."""
        self.enable_timer_groups()
        env_cfg, env = self.instantiate_env_cfg()


        # counter for number of frames to run for
        num_frames = 0
        # log frame times
        while simulation_app.is_running():
            while num_frames < num_frames:
                # get upper and lower bounds of action space, sample actions randomly on this interval
                action_high = 1
                action_low = -1
                actions = (action_high - action_low) * torch.rand(
                    env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device
                ) - action_high

                # env stepping
                _ = env.step(actions)

                num_frames += 1

            # terminate
            break
        self.store_measurements()
        print("Done benchmarking!")
        env.close()

    def store_measurements(self):
        timings = Timer.timing_info
        for group in _class_group_registry.keys():
            for cls in list(_class_group_registry.get(group, ())):
                try:
                    for name in getattr(cls, "_class_timer_mappings", {}).keys():
                        uname = getattr(cls, "_class_timer_mappings", {})[name]
                        if timings[uname]["n"] > 0:
                            self.refined_timings[f"{cls.__name__}.{name}"] = {}
                            self.refined_timings[f"{cls.__name__}.{name}"]["mean"] = timings[uname]["mean"]
                            self.refined_timings[f"{cls.__name__}.{name}"]["std"] = timings[uname]["std"]
                            self.refined_timings[f"{cls.__name__}.{name}"]["n"] = timings[uname]["n"]
                except Exception as e:
                    print(f"{cls.__name__}: {e}")

class ASVWrapper:
    data = {} 
    number = 1
    rounds = 1
    sample_time = 0
    repeat = 1
    replay = ""

    params = ()
    param_names = ("mean")

    def setup_cache(self):
        return 


if __name__ == "__main__":
    # run the main function
    cartpole_benchmark = SuiteNonRLBenchmark(task="Isaac-Cartpole-Direct-v0", num_envs=4096, seed=1, num_frames=10000, args=args)
    cartpole_benchmark.run()


    # close sim app
    simulation_app.close()
