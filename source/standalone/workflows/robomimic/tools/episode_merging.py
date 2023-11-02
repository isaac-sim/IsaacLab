# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool to merge multiple episodes with single trajectory into one episode with multiple trajectories."""

from __future__ import annotations

import argparse
import h5py
import json
import os

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Merge multiple episodes with single trajectory into one episode.")
    parser.add_argument(
        "--dir", type=str, default=None, help="Path to directory that contains all single episode hdf5 files"
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--out", type=str, default="merged_dataset.hdf5", help="output hdf5 file")
    args_cli = parser.parse_args()

    # read arguments
    parent_dir = args_cli.dir
    merged_dataset_name = args_cli.out
    task_name = args_cli.task
    # check valid task name
    if task_name is None:
        raise ValueError("Please specify a valid task name.")

    # get hdf5 entries from specified directory
    entries = [i for i in os.listdir(parent_dir) if i.endswith(".hdf5")]

    # create new hdf5 file for merging episodes
    fp = h5py.File(parent_dir + merged_dataset_name, "a")

    # initiate data group
    f_grp = fp.create_group("data")
    f_grp.attrs["num_samples"] = 0

    # merge all episodes
    for count, entry in enumerate(entries):
        fc = h5py.File(parent_dir + entry, "r")

        # find total number of samples in all demos
        f_grp.attrs["num_samples"] = f_grp.attrs["num_samples"] + fc["data"]["demo_0"].attrs["num_samples"]

        fc.copy("data/demo_0", fp["data"], "demo_" + str(count))

    # This is needed to run env in robomimic
    fp["data"].attrs["env_args"] = json.dumps({"env_name": task_name, "type": 2, "env_kwargs": {}})

    fp.close()

    print("merged")
