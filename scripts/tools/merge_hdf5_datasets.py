# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

import h5py

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

    # A single root-level format_version applies to every episode in the output.
    # Treat a missing attribute as the legacy version 0 and reject heterogeneous
    # inputs before creating the output file so episodes cannot be mislabeled.
    format_versions = []
    for filepath in args_cli.input_files:
        with h5py.File(filepath, "r") as input:
            format_versions.append(int(input.attrs.get("format_version", 0)))
    if len(set(format_versions)) > 1:
        versions = ", ".join(
            f"{filepath}={version}" for filepath, version in zip(args_cli.input_files, format_versions)
        )
        raise ValueError(f"Cannot merge datasets with different format_version values: {versions}.")

    with h5py.File(args_cli.output_file, "w") as output:
        episode_idx = 0
        copy_attributes = True

        for filepath in args_cli.input_files:
            with h5py.File(filepath, "r") as input:
                for episode, data in input["data"].items():
                    input.copy(f"data/{episode}", output, f"data/demo_{episode_idx}")
                    episode_idx += 1

                if copy_attributes:
                    if "format_version" in input.attrs:
                        output.attrs["format_version"] = input.attrs["format_version"]
                    output["data"].attrs["env_args"] = input["data"].attrs["env_args"]
                    copy_attributes = False

    print(f"Merged dataset saved to {args_cli.output_file}")


if __name__ == "__main__":
    merge_datasets()
