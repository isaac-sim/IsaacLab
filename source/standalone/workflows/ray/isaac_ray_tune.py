# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# # Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# # import isaac_ray_util
# # import ray
# # from ray import tune


# class JobCfg:
#     def __init__(self, cfg):
#         assert "runner_args" in cfg, "No runner arguments specified"
#         assert "workflow" in cfg, "No workflow specified"
#         assert "hydra_args" in cfg, "No hypeparameters specified"


# class RLGamesCameraJobCfg(JobCfg):
#     def __init__(self, cfg: dict):
#         # Modify cfg to include headless and enable_cameras
#         cfg["runner_args"]["singletons"] = []
#         cfg["runner_args"]["singletons"].append("--headless")
#         cfg["runner_args"]["singletons"].append("--enable_cameras")
#         cfg["workflow"] = "/workspace/isaaclab/workflows/rl_games/train.py"
#         super().__init__(cfg)


# class RLGamesCameraJobCfgHelper(RLGamesCameraJobCfg):
#     """ """

#     def __init__(self, cfg={}, vary_env_count: bool = True, vary_cnn: bool = False, vary_mlp: bool = True):
#         cfg["hydra_args"]["agent.params.config.save_best_after"] = 5
#         cfg["hydra_args"]["agent.params.config.save_frequency"] = 5

#         if vary_env_count:
#             """
#             Ideally, more envs are better. However, this may not actually be the case.
#             """

#             def batch_size_divisors(batch_size):
#                 return [i for i in range(1, batch_size + 1) if batch_size % i == 0]

#             cfg["runner_args"]["--num_envs"] = tune.randint(2**6, 2**14)
#             cfg["hydra_args"]["agent.params.horizon_length"] = tune.randint(1, 200)
#             cfg["hydra_args"]["minibatch_size"] = (
#                 tune.sample_from(
#                     lambda spec: tune.choice(
#                         batch_size_divisors(spec.config.horizon_length * spec.config.num_envs * spec.config.num_envs)
#                     )
#                 ),
#             )

#         super().__init__(cfg)


# class RLGamesResNetCameraJob(RLGamesCameraJobCfgHelper):
#     def __init__(self, cfg: dict = {}):
#         cfg["hydra_args"]["env.observations.policy.image.params.model_name"] = tune.choice(
#             ["resnet18", "resnet34", "resnet50", "resnet101"]
#         )
#         super().__init__(cfg, vary_env_count=True, vary_cnn=False, vary_mlp=True)


# class RLGamesTheiaCameraJob(RLGamesCameraJobCfgHelper):
#     def __init__(self, cfg: dict = {}):
#         cfg["hydra_args"]["env.observations.policy.image.params.model_name"] = tune.choice([
#             "theia-tiny-patch16-224-cddsv",
#             "theia-tiny-patch16-224-cdiv",
#             "theia-small-patch16-224-cdiv",
#             "theia-base-patch16-224-cdiv",
#             "theia-small-patch16-224-cddsv",
#             "theia-base-patch16-224-cddsv",
#         ])
#         super().__init__(cfg, vary_env_count=True, vary_cnn=False, vary_mlp=True)


# class IsaacLabTuneTrainable(tune.Trainable):
#     def __init__(self):
#         pass
#         # self.invocation_str = executable_path + " " + workflow_path
#         # for arg in args:
#         #     spaced_arg = " " + arg + " "
#         #     self.invocation_str += spaced_arg
#         # print(f"[INFO] Using base invocation of {self.invocation_str} for all trials")

#     def setup(self, config):
#         pass

#     def step(self):
#         pass
