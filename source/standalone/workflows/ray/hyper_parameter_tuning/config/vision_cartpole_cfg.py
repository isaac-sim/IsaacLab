# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# # Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# # import pathlib
# # import sys

# # from ray import tune

# # UTIL_DIR = pathlib.Path(__file__).parent.parent.parent
# # sys.path.append(str(UTIL_DIR))

# # import isaac_ray_tune
# # import isaac_ray_util


# class CartpoleRGB(isaac_ray_tune.RLGamesCameraJobCfgHelper):
#     def __init__(self, cfg: dict = {}):
#         cfg["runner_args"]["--task"] = "Isaac-Cartpole-RGB-Camera-v0"
#         super().__init__(cfg, vary_env_count=True, vary_cnn=True, vary_mlp=True)


# class CartpoleResNet(isaac_ray_tune.RLGamesResNetCameraJob):
#     def __init__(self, cfg: dict = {}):
#         cfg["runner_args"]["--task"] = "Isaac-Cartpole-ResNet18-Camera-v0"
#         super().__init__(cfg)


# class CartpoleTheia(isaac_ray_tune.RLGamesTheiaCameraJob):
#     def __init__(self, cfg: dict = {}):
#         cfg["runner_args"]["--task"] = "Isaac-Cartpole-TheiaTiny-Camera-v0"
#         super().__init__(cfg)
