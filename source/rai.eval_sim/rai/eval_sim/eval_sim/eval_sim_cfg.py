# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import enum
import os
import stat

import yaml
from isaaclab.utils import configclass
from rai.eval_sim.utils import USER_EVAL_SIM_CFG_PATH


class ExecutionMode(enum.IntEnum):
    """Whether to run EvalSim in a lockstep fashion, or asynchronously."""

    BLOCKING: int = 0
    NON_BLOCKING: int = 1


@configclass
class EvalSimCfg:
    """Configuration class for EvalSim.

    NOTE: This class has a corresponding yaml file that is used to save and load configurations, so that each
    user can have their own configuration settings. The yaml file is located at USER_EVAL_SIM_CFG_PATH.
    """

    # general settings
    execution_mode: ExecutionMode = ExecutionMode.BLOCKING  # currently only blocking mode is supported
    control_delay: int = 0  # delay in control commands
    sync_to_real_time: bool = False  # limit sim execution to be no faster than real time

    # environment / ros manager settings
    env_cfg: str = ""  # the environment configuration file to use
    ros_manager_cfg: str = ""  # the ROS Manager configuration file to use
    auto_load_env: bool = True  # whether to auto load the environment and ros_manager configs upon startup
    auto_save: bool = True  # whether to auto save the configurations in YAML upon loading them
    enable_ros: bool = False  # whether to enable ROS upon startup

    # profiling
    wallclock_dt_buffer_size: int = 200

    # importing environment / ros manager configurations
    search_pkgs_for_cfgs: list = [
        # add packages from which you would like to import environment or ros_manager cfg's
        # NOTE: these packages need to be pip installed to be used.
        "rai.eval_sim",
        "rai.eval_sim_cfgs",
        # "rai.humanoid",
        # "rai.spot",
        # "rai.umv",
        # "rai.franka",
        # "isaaclab_tasks",  # NOTE: contains a lot of envs, you might not want them all to be loaded.
    ]

    def __str__(self):
        """Return the config in a nicely formatted string for printing."""
        # Need to modify execution_mode to be a string for printing
        as_dict = copy.deepcopy(self.__dict__)
        as_dict["execution_mode"] = self.execution_mode.name
        return yaml.dump(as_dict, default_flow_style=False, sort_keys=False)

    def to_yaml(self, file_path: str = USER_EVAL_SIM_CFG_PATH):
        """Save the EvalSimCfg object to a yaml file.

        Args:
            file_path: Path to the yaml file to save to. Defaults to USER_EVAL_SIM_CFG_PATH.
        """

        with open(file_path, "w") as f:
            as_dict = copy.deepcopy(self.__dict__)
            # Execution mode is an enum, so we need to convert it to a string
            as_dict["execution_mode"] = self.execution_mode.name
            yaml.dump(as_dict, f, default_flow_style=False, sort_keys=False)

        os.chmod(file_path, stat.S_IRWXO)

    @classmethod
    def from_yaml(cls, file_path: str = USER_EVAL_SIM_CFG_PATH):
        """Create an EvalSimCfg object from a yaml file.

        Args:
            file_path: Path to the yaml file to load from. Defaults to USER_EVAL_SIM_CFG_PATH.

        Returns:
            EvalSimCfg: The EvalSimCfg object created from the yaml file.

        """
        instance = cls()
        unknown_keys = []  # list of keys that are not part of the EvalSimCfg class
        # handle the case where the file doesn't exist
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Using default configuration.")
            instance.to_yaml(file_path)
            return instance

        # TODO: How does this handle unknown keys? Should raise an exception with a helpful message.
        with open(file_path) as f:
            data = yaml.safe_load(f)
            for key, value in data.items():
                if key == "execution_mode":
                    value = ExecutionMode[value]
                if hasattr(instance, key):
                    setattr(instance, key, value)
                else:
                    unknown_keys.append(key)

        if len(unknown_keys) > 0:
            raise ValueError(
                f"The following key(s) {unknown_keys} in the config file {file_path} are not part of the EvalSimCfg "
                "class, remove them or delete the config file to use the default configuration."
            )

        return instance
