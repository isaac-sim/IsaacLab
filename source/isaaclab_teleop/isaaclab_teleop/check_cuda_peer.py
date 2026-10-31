# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check synchronized peer-copy values before selecting split-GPU teleoperation.

Run as ``python -m isaaclab_teleop.check_cuda_peer --devices cuda:0 cuda:1``.
This is a small transfer-integrity check, not an application or performance test.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack

# A 16 KiB payload with exactly representable float32 values, distinct from zero.
_VALUE_COUNT = 4096


def main() -> int:
    """Check two patterns in both directions; return nonzero on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs=2, required=True, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()

    import numpy as np
    import warp as wp

    try:
        wp.init()
        devices = [wp.get_device(name) for name in args.devices]
        if not all(device.is_cuda for device in devices) or devices[0] == devices[1]:
            raise ValueError("Select two distinct CUDA devices.")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
        for device in devices:
            print(f"{device}: {device.name}; UUID={device.uuid}; PCI={device.pci_bus_id}")

        passed = True
        with ExitStack() as scope:
            for device in devices:
                # Exercise direct peer access with ordinary CUDA allocations.
                scope.enter_context(wp.ScopedMempool(device, enable=False))
            for source, target in (devices, devices[::-1]):
                if not wp.is_peer_access_supported(source, target):
                    raise RuntimeError(f"Direct peer access is unsupported: {source} -> {target}.")
                scope.enter_context(wp.ScopedPeerAccess(source, target, enable=True))
            for source, target in (devices, devices[::-1]):
                stream = wp.Stream(target)
                for sign in (1, -1):
                    expected = np.arange(1, _VALUE_COUNT + 1, dtype=np.float32) * sign
                    source_values = wp.array(expected, device=source)
                    target_values = wp.zeros(_VALUE_COUNT, dtype=wp.float32, device=target)
                    # Initialize both allocations before the foreign-device copy.
                    wp.synchronize_device(source)
                    wp.synchronize_device(target)
                    wp.copy(target_values, source_values, stream=stream)
                    wp.synchronize_stream(stream)
                    wrong = int(np.count_nonzero(target_values.numpy() != expected))
                    passed = passed and wrong == 0
                    print(f"{source} -> {target}, sign={sign:+d}: {wrong}/{_VALUE_COUNT} incorrect values")
        if passed:
            print("PASS: all four peer-copy value comparisons are exact.")
            return 0
    except (RuntimeError, ValueError) as error:
        print(f"Peer-copy check failed: {error}", file=sys.stderr)

    print(
        "Do not use this split-GPU path until the failure is understood. Use one physical GPU, or ask an"
        " administrator to check PCIe peer access and IOMMU translation with idle GPUs. This command does"
        " not change drivers, host services or IOMMU settings.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
