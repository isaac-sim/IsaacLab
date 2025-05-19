# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from rai.eval_sim.utils import TASKS_DIR

from .testing import TESTING_CFGS
from .testing.anymal_d_testing_env_cfg import AnymalDTestAllEnvCfg

# Dictionary mapping task names to a tuple containing their environment and ROS manager configurations.
PER_ROBOT_EVAL_SIM_CFGS = {
    "anymal": (
        "rai.eval_sim.tasks.anymal.anymal_env_cfg.AnymalDEnvCfg",
        "rai.eval_sim.tasks.anymal.anymal_ros_manager_cfg.AnymalDRosManagerCfg",
    ),
    "franka": (
        "rai.eval_sim.tasks.franka.franka_env_cfg.FrankaEnvCfg",
        "rai.eval_sim.tasks.franka.franka_ros_manager_cfg.FrankaRosManagerCfg",
    ),
    "single_drive": (
        "rai.eval_sim.tasks.single_drive.single_drive_env_cfg.SingleDriveEnvCfg",
        "rai.eval_sim.tasks.single_drive.single_drive_ros_manager_cfg.SingleDriveRosManagerCfg",
    ),
    "spot": (
        "rai.eval_sim.tasks.spot.spot_env_cfg.SpotEnvCfg",
        "rai.eval_sim.tasks.spot.spot_env_cfg.SpotRosManagerCfg",
    ),
    "anymal_camera": (
        "rai.eval_sim.tasks.anymal_camera.anymal_camera_env_cfg.AnymalDCameraEnvCfg",
        "rai.eval_sim.tasks.anymal_camera.anymal_camera_ros_manager_cfg.AnymalDCameraRosManagerCfg",
    ),
}
