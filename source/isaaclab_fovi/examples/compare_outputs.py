# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare RGB snapshots saved by render_foveated.py across installations."""

import argparse
import statistics

import torch


def main() -> None:
    """Report image differences and observed frame times without setting thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if reference["rgb"].shape != candidate["rgb"].shape:
        raise ValueError("RGB shapes differ")
    if reference["rgb"].dtype != candidate["rgb"].dtype:
        raise ValueError("RGB dtypes differ")
    torch.testing.assert_close(reference["sampling_coordinates"], candidate["sampling_coordinates"])
    difference = (reference["rgb"].float() - candidate["rgb"].float()).abs()
    print(f"RGB absolute error: mean={difference.mean().item()}, max={difference.max().item()}")
    for label, snapshot in (("reference", reference), ("candidate", candidate)):
        durations = snapshot["frame_seconds"]
        if durations:
            print(f"{label} median warm frame seconds: {statistics.median(durations)}")


if __name__ == "__main__":
    main()
