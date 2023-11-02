# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Script for splitting a dataset hdf5 file into training and validation trajectories.

Args:
    dataset: path to hdf5 dataset

    filter_key: if provided, split the subset of trajectories
        in the file that correspond to this filter key into a training
        and validation set of trajectories, instead of splitting the
        full set of trajectories

    ratio: validation ratio, in (0, 1). Defaults to 0.1, which is 10%.

Example usage:
    python split_train_val.py --dataset /path/to/demo.hdf5 --ratio 0.1
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np

from robomimic.utils.file_utils import create_hdf5_filter_key


def split_train_val_from_hdf5(hdf5_path: str, val_ratio=0.1, filter_key=None):
    """
    Splits data into training set and validation set from HDF5 file.

    Args:
        hdf5_path: path to the hdf5 file to load the transitions from
        val_ratio: ratio of validation demonstrations to all demonstrations

        filter_key: if provided, split the subset of demonstration keys stored
            under mask/@filter_key instead of the full set of demonstrations
    """
    # retrieve number of demos
    f = h5py.File(hdf5_path, "r")
    if filter_key is not None:
        print(f"Using filter key: {filter_key}")
        demos = sorted(elem.decode("utf-8") for elem in np.array(f[f"mask/{filter_key}"]))
    else:
        demos = sorted(list(f["data"].keys()))
    num_demos = len(demos)
    f.close()

    # get random split
    num_demos = len(demos)
    num_val = int(val_ratio * num_demos)
    mask = np.zeros(num_demos)
    mask[:num_val] = 1.0
    np.random.shuffle(mask)
    mask = mask.astype(int)
    train_inds = (1 - mask).nonzero()[0]
    valid_inds = mask.nonzero()[0]
    train_keys = [demos[i] for i in train_inds]
    valid_keys = [demos[i] for i in valid_inds]
    print(f"{num_val} validation demonstrations out of {num_demos} total demonstrations.")

    # pass mask to generate split
    name_1 = "train"
    name_2 = "valid"
    if filter_key is not None:
        name_1 = f"{filter_key}_{name_1}"
        name_2 = f"{filter_key}_{name_2}"

    train_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=train_keys, key_name=name_1)
    valid_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=valid_keys, key_name=name_2)

    print(f"Total number of train samples: {np.sum(train_lengths)}")
    print(f"Average number of train samples {np.mean(train_lengths)}")

    print(f"Total number of valid samples: {np.sum(valid_lengths)}")
    print(f"Average number of valid samples {np.mean(valid_lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help=(
            "If provided, split the subset of trajectories in the file that correspond to this filter key"
            " into a training and validation set of trajectories, instead of splitting the full set of"
            " trajectories."
        ),
    )
    parser.add_argument("--ratio", type=float, default=0.1, help="validation ratio, in (0, 1)")
    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    split_train_val_from_hdf5(args.dataset, val_ratio=args.ratio, filter_key=args.filter_key)
