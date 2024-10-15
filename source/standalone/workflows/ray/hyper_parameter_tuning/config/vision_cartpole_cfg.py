# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import pathlib
import sys

UTIL_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(UTIL_DIR))

import isaac_ray_tune
import isaac_ray_util


class CartpoleRGBNoTuneJobCfg(isaac_ray_tune.JobCfg):  # Idempotent
    def __init__(self, cfg: dict = {}):
        cfg["workflow"] = "/workspace/isaaclab/workflows/rl_games/train.py"
        cfg["runner_args"]["singletons"] = ["--headless", "--enable_cameras"]
        cfg["hydra_args"] = {}
        super().__init__(cfg)


class CartpoleRGBJobCfg(isaac_ray_tune.RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-RGB-Camera-v0"
        super().__init__(cfg, vary_env_count=True, vary_cnn=True, vary_mlp=True)


class CartpoleResNetJobCfg(isaac_ray_tune.RLGamesResNetCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-ResNet18-Camera-v0"
        super().__init__(cfg)


class CartpoleTheiaJobCfg(isaac_ray_tune.RLGamesTheiaCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-TheiaTiny-Camera-v0"
        super().__init__(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Cartpole.")
    parser.add_argument("--tune_type", choices=["standard_no_tune", "standard", "resnet", "theia"])
    isaac_ray_util.add_cluster_args(parser=parser)
    args = parser.parse_args()
    cfg = None
    if args.type == "standard_no_tune":
        cfg = CartpoleRGBNoTuneJobCfg()
    elif args.type == "standard":
        cfg = CartpoleRGBJobCfg()
    elif args.type == "resnet":
        cfg = CartpoleResNetJobCfg()
    elif args.type == "theia":
        cfg = CartpoleTheiaJobCfg()
