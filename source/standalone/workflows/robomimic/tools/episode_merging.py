# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tool to merge multiple episodes with single trajectory into one episode with multiple trajectories.
"""


import argparse
import h5py
import json
import os

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
    parser.add_argument(
        "--dir", type=str, default=None, help="PATH to directory that contains all single episode hdf5 files"
    )
    parser.add_argument("--out", type=str, default="merged_dataset.hdf5", help="output hdf5 file")
    args_cli = parser.parse_args()

    parent_dir = args_cli.dir
    merged_dataset_name = args_cli.out

    # get hdf5 entries from specified directory
    entries = [i for i in os.listdir(parent_dir) if i.endswith(".hdf5")]

    # create new hdf5 file for merging episodes
    fp = h5py.File(parent_dir + merged_dataset_name, "a")

    # init
    f_grp = fp.create_group("data")
    f_grp.attrs["num_samples"] = 0
    count = 0

    for entry in entries:
        fc = h5py.File(parent_dir + entry, "r")

        # find total number of samples in all demos
        f_grp.attrs["num_samples"] = f_grp.attrs["num_samples"] + fc["data"]["demo_0"].attrs["num_samples"]

        fc.copy("data/demo_0", fp["data"], "demo_" + str(count))
        count += 1

    # This is needed to run env in robomimic
    fp["data"].attrs["env_args"] = json.dumps({"env_name": "Isaac-Lift-v0", "type": 2, "env_kwargs": {}})

    fp.close()

    print("merged")
