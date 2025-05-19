# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to launch EvalSim in standalone mode.

Example:
# to launch the builtin AnymalDEnvCfg and AnymalDRosManagerCfg from rai.eval_sim
python standalone.py --robot anymal

or

# to launch any configs in the python workspace
python standalone.py --env_cfg rai.eval_sim_cfgs.umv.mini_env_cfg.MiniEnvCfg --ros_cfg rai.eval_sim_cfgs.umv.mini_ros_manager_cfg.MiniRosManagerCfg

or

# to launch using eval_sim_cfg.yaml file
python standalone.py

    # in eval_sim_cfg.yaml (found at EvalSimCfg.USER_EVAL_SIM_CFG_PATH)
    # file set the 'env_cfg' and 'ros_manager_cfg' to the desired import path, e.g. for UMV Mini:
    env_cfg: rai.eval_sim_cfgs.umv.mini_env_cfg.MiniEnvCfg
    ros_manager_cfg: rai.eval_sim_cfgs.umv.mini_ros_manager_cfg.MiniRosManagerCfg
"""
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

# update and parse AppLauncher arguments
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="EvalSim Standalone")  # add argparse arguments
parser.add_argument(
    "--robot",
    type=str,
    choices=["anymal", "franka", "single_drive", "spot", "anymal_camera"],
    help=(
        "Example Robot to load ManagerBasedEnvCfg and RosManagerCfg for. For other robots please use 'env_cfg' and "
        "'ros_cfg' arguments."
    ),
)
parser.add_argument(
    "--sync_to_real_time",
    action="store_true",
    default=False,
    help="Limit sim execution to be no faster than real time.",
)
parser.add_argument(
    "--env_cfg", type=str, default="", help="The python package import path for the ManagerBasedEnvCfg of choice."
)
parser.add_argument(
    "--ros_cfg", type=str, default="", help="The python package import path for the RosManagerCfg of choice."
)
parser.add_argument(
    "--ignore_user_cfg",
    action="store_true",
    default=False,
    help=(
        "If set, ignores the user_eval_sim_cfg.yaml file and uses only the provided arguments on top of defaults."
        " If not set the user_eval_sim_cfg.yaml file will be used and overwritten with any provided arguments."
    ),
)
AppLauncher.add_app_launcher_args(parser)  # append AppLauncher cli args
args_cli = parser.parse_args()  # parse the arguments

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback

import carb
from rai.eval_sim.eval_sim import EvalSimCfg, EvalSimStandalone
from rai.eval_sim.tasks import PER_ROBOT_EVAL_SIM_CFGS
from rai.eval_sim.utils import USER_EVAL_SIM_CFG_PATH, log_warn


def main():
    if args_cli.ignore_user_cfg:
        log_warn("Ignoring user_eval_sim_cfg.yaml file and using only the provided arguments on top of defaults.")
        eval_sim_cfg = EvalSimCfg()
    else:
        eval_sim_cfg = EvalSimCfg.from_yaml()

    eval_sim_cfg.sync_to_real_time = args_cli.sync_to_real_time

    # load EvalSim
    if args_cli.robot is None:
        if args_cli.ros_cfg == "" or args_cli.env_cfg == "":
            log_warn(
                "If 'robot' is not set and 'env_cfg' and 'ros_cfg' are not set, "
                "configs will be based upon the 'user_eval_sim_cfg.yaml' file."
                f"(Full path: {USER_EVAL_SIM_CFG_PATH})"
            )

        # update the eval_sim_cfg with the env and ros_manager cfgs from the CLI
        eval_sim_cfg.env_cfg = args_cli.env_cfg
        eval_sim_cfg.ros_manager_cfg = args_cli.ros_cfg

    else:
        # check to make sure custom env_cfg and ros_cfg are left as default
        assert args_cli.ros_cfg == "", "CLI argument 'ros_cfg' cannot be set when 'robot' is set"
        assert args_cli.env_cfg == "", "CLI argument 'env_cfg' cannot be set when 'robot' is set"

        # check for valid builtin robot
        try:
            per_robot_eval_sim_path_cfg = PER_ROBOT_EVAL_SIM_CFGS[args_cli.robot]
        except KeyError:
            raise ValueError(f"Invalid robot: {args_cli.robot}. Valid robots are: {PER_ROBOT_EVAL_SIM_CFGS.keys()}")

        # update the eval_sim_cfg with the env and ros_manager cfgs from the per_robot_eval_sim_path_cfg
        eval_sim_cfg.env_cfg = str(per_robot_eval_sim_path_cfg[0])
        eval_sim_cfg.ros_manager_cfg = str(per_robot_eval_sim_path_cfg[1])

    # create eval_sim instance
    eval_sim = EvalSimStandalone(eval_sim_cfg)

    # run eval_sim standalone simulation
    while simulation_app.is_running():
        eval_sim.step()

    eval_sim.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
