# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for sensor base."""

import warp as wp


@wp.kernel
def update_timestamp_kernel(
    is_outdated: wp.array(dtype=wp.bool),
    timestamp: wp.array(dtype=wp.float32),
    timestamp_last_update: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    update_period: wp.float32,
):
    """Updates timestamp and marks environments as outdated if update period elapsed.

    Args:
        is_outdated: Boolean array indicating which envs need update.
        timestamp: Current timestamp per env.
        timestamp_last_update: Last update timestamp per env.
        dt: Time step per env.
        update_period: Period after which sensor should be updated.
    """
    env = wp.tid()
    new_timestamp = timestamp[env] + dt[env]
    timestamp[env] = new_timestamp
    if new_timestamp - timestamp_last_update[env] + 1e-6 >= update_period:
        is_outdated[env] = True


@wp.kernel
def update_outdated_envs_kernel(
    is_outdated: wp.array(dtype=wp.bool),
    timestamp: wp.array(dtype=wp.float32),
    timestamp_last_update: wp.array(dtype=wp.float32),
):
    """Updates timestamp and clears outdated flag for outdated environments.

    Args:
        is_outdated: Boolean array indicating which envs need update. Will be set to False.
        timestamp: Current timestamp per env.
        timestamp_last_update: Last update timestamp per env. Will be set to current timestamp.
    """
    env = wp.tid()
    if is_outdated[env]:
        timestamp_last_update[env] = timestamp[env]
        is_outdated[env] = False


@wp.kernel
def reset_envs_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    is_outdated: wp.array(dtype=wp.bool),
    timestamp: wp.array(dtype=wp.float32),
    timestamp_last_update: wp.array(dtype=wp.float32),
):
    """Resets the current and last update timestamps and marks environments as outdated for those being reset.

    Args:
        is_outdated: Boolean array indicating which envs need update. Will be set to False.
        timestamp: Current timestamp per env.
        timestamp_last_update: Last update timestamp per env. Will be set to current timestamp.
    """

    env = wp.tid()
    if not reset_mask[env]:
        return

    # Reset the timestamp for the sensors
    timestamp[env] = 0.0

    timestamp_last_update[env] = 0.0
    # Set all reset sensors to outdated so that they are updated when data is called the next time.
    is_outdated[env] = True
