# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedEnvCfg


def update_env_cfg(cfg: ManagerBasedEnvCfg):
    """Update an environment config to ensure it works and performs well with EvalSim.

    These modifications are necessary so that raw ManagerBasedEnvcfg and ManagerBasedRLEnvCfgs can be used seamlessly with eval sim.
    Most settings ensure higher performance over default settings, which is crucial for a good user experience.
    NOTE: The input config is modified in place.

    Args:
        cfg: The config to update.
    """

    # general adjustments
    cfg.scene.num_envs = 1
    cfg.ui_window_class_type = None

    # cpu deployment ensures higher performance over gpu when parallelization is not required
    cfg.sim.device = "cpu"
    cfg.sim.use_gpu_pipeline = False
    cfg.sim.physx.use_gpu = False

    # enable fabric
    cfg.sim.use_fabric = True

    # ensures enhanced determinism without much loss in performance (only available for cpu)
    cfg.sim.physx.enable_enhanced_determinism = True
