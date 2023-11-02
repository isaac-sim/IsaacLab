# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool to check structure of hdf5 files."""

from __future__ import annotations

import argparse
import h5py


def check_group(f, num: int):
    """Print the data from different keys in stored dictionary."""
    # print name of the group first
    for subs in f:
        if isinstance(subs, str):
            print("\t" * num, subs, ":", type(f[subs]))
            check_group(f[subs], num + 1)
    # print attributes of the group
    print("\t" * num, "attributes", ":")
    for attr in f.attrs:
        print("\t" * (num + 1), attr, ":", type(f.attrs[attr]), ":", f.attrs[attr])


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Check structure of hdf5 file.")
    parser.add_argument("file", type=str, default=None, help="The path to HDF5 file to analyze.")
    args_cli = parser.parse_args()

    # open specified file
    with h5py.File(args_cli.file, "r") as f:
        # print name of the file first
        print(f)
        # print contents of file
        check_group(f["data"], 1)
