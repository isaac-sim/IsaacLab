# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import h5py
import os

parser = argparse.ArgumentParser(description="Merge a set of HDF5 datasets.")
parser.add_argument(
    "--input_files",
    type=str,
    nargs="+",
    default=[],
    help="A list of paths to HDF5 files to merge.",
)
parser.add_argument("--output_file", type=str, default="merged_dataset.hdf5", help="File path to merged output.")

args_cli = parser.parse_args()


def merge_datasets():
    for filepath in args_cli.input_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The dataset file {filepath} does not exist.")

    with h5py.File(args_cli.output_file, "w") as output:
        episode_idx = 0
        copy_attributes = True

        for filepath in args_cli.input_files:

            with h5py.File(filepath, "r") as input:
                for episode, data in input["data"].items():
                    input.copy(f"data/{episode}", output, f"data/demo_{episode_idx}")
                    episode_idx += 1

                if copy_attributes:
                    output["data"].attrs["env_args"] = input["data"].attrs["env_args"]
                    copy_attributes = False

    print(f"Merged dataset saved to {args_cli.output_file}")


if __name__ == "__main__":
    merge_datasets()
