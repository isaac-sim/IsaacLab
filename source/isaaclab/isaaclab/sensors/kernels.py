# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def update_timestamp_kernel(
    is_outdated: wp.array(dtype=wp.bool),
    timestamp: wp.array(dtype=wp.float32),
    timestamp_last_update: wp.array(dtype=wp.float32),
    dt: wp.float32,
    update_period: wp.float32,
):
    """Updates timestamp and marks environments as outdated if update period elapsed.

    Args:
        is_outdated: Boolean array indicating which envs need update.
        timestamp: Current timestamp per env.
        timestamp_last_update: Last update timestamp per env.
        dt: Simulation time step (scalar).
        update_period: Period after which sensor should be updated.
    """
    env = wp.tid()
    new_timestamp = timestamp[env] + dt
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
    """Resets timestamps and clears the outdated flag for reset environments.

    The outdated flag is cleared (not set) so that a sensor read immediately after
    :meth:`SensorBase.reset` returns whatever the subclass wrote into ``_data`` during
    reset (typically zero), rather than triggering a refetch from a physics buffer that
    has not been stepped since the reset. The next call to :func:`update_timestamp_kernel`
    (driven by :meth:`InteractiveScene.update` after the next physics step) re-arms the
    outdated flag, at which point the sensor will pull fresh post-reset values.

    Args:
        reset_mask: Boolean array indicating which envs to reset.
        is_outdated: Boolean array indicating which envs need update. Will be cleared to False for reset envs.
        timestamp: Current timestamp per env. Will be set to 0.0 for reset envs.
        timestamp_last_update: Last update timestamp per env. Will be set to 0.0 for reset envs.
    """

    env = wp.tid()
    if not reset_mask[env]:
        return

    timestamp[env] = 0.0
    timestamp_last_update[env] = 0.0
    is_outdated[env] = False
