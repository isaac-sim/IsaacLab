# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import h5py


def write_log_to_hdf5(held_asset_pose_log, fixed_asset_pose_log, success_log, eval_logging_filename):

    with h5py.File(eval_logging_filename, "w") as hf:
        hf.create_dataset("held_asset_pose", data=held_asset_pose_log.cpu().numpy())
        hf.create_dataset("fixed_asset_pose", data=fixed_asset_pose_log.cpu().numpy())
        hf.create_dataset("success", data=success_log.cpu().numpy())


def load_log_from_hdf5(eval_logging_filename):

    with h5py.File(eval_logging_filename, "r") as hf:
        held_asset_pose = hf["held_asset_pose"][:]
        fixed_asset_pose = hf["fixed_asset_pose"][:]
        success = hf["success"][:]

    # held_asset_pose = torch.from_numpy(held_asset_pose).to(device)
    # fixed_asset_pose = torch.from_numpy(fixed_asset_pose).to(device)
    # success = torch.from_numpy(success).to(device)

    return held_asset_pose, fixed_asset_pose, success
