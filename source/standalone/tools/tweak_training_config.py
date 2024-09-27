# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script can help you

.. code-block:: bash
    # Usage with GUI
    ./isaaclab.sh -p source/standalone/tools/-h

    # Usage with headless
    ./isaaclab.sh -p source/standalone/tools/ -h --headless
"""

import argparse

# parse the arguments
args_cli = argparse.Namespace()

parser = argparse.ArgumentParser(description="This script can help you tweak run parameters.")

test = {
    "rgb-cam-sweep": {
        "task_name": "Isaac-Cartpole-RGB-Camera-v0",
        # relative cfg path is hardcoded as opposed to loaded from the registry so that
        # we don't have to launch an applauncher instance here, and can instead wait to do that in the trainer
        "relative_cfg_path": (
            "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/cartpole"
        ),
        # training tool path is hardcoded as it is not available as a self contained import
        "relative_training_tool_path": "source/standalone/workflows/rl_games/train.py",
        "relative_log_output_path_regex": "source/standalone/workflows/rl_games/logs",
    }
}
