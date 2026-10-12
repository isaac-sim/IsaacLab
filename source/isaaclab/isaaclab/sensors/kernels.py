# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def update_timestamp_kernel(
    is_outdated: wp.array(dtype=wp.bool),
    timestamp: wp.array(dtype=wp.float32),
    elapsed_since_update: wp.array(dtype=wp.float64),
    dt: wp.float64,
    due_threshold: wp.float64,
):
    """Advances the sensor clocks and marks environments as outdated if the update period elapsed.

    Due-ness is decided on the time elapsed since the last refresh, accumulated in float64 from zero.
    Subtracting two float32 timestamps instead loses precision as the timestamps grow, which delays
    refreshes once the episode has run for some seconds.

    Args:
        is_outdated: Boolean array indicating which envs need update.
        timestamp: Current timestamp per env [s].
        elapsed_since_update: Time since the last refresh per env [s].
        dt: Simulation time step [s].
        due_threshold: Elapsed time at which the sensor is due, ``update_period - 1e-6`` [s].
    """
    env = wp.tid()
    timestamp[env] = timestamp[env] + wp.float32(dt)
    elapsed = elapsed_since_update[env] + dt
    elapsed_since_update[env] = elapsed
    if elapsed >= due_threshold:
        is_outdated[env] = True


@wp.kernel
def update_outdated_envs_kernel(
    is_outdated: wp.array(dtype=wp.bool),
    timestamp: wp.array(dtype=wp.float32),
    timestamp_last_update: wp.array(dtype=wp.float32),
    elapsed_since_update: wp.array(dtype=wp.float64),
):
    """Updates timestamp and clears outdated flag for outdated environments.

    Args:
        is_outdated: Boolean array indicating which envs need update. Will be set to False.
        timestamp: Current timestamp per env.
        timestamp_last_update: Last update timestamp per env. Will be set to current timestamp.
        elapsed_since_update: Time since the last refresh per env. Will be set to 0.0.
    """
    env = wp.tid()
    if is_outdated[env]:
        timestamp_last_update[env] = timestamp[env]
        elapsed_since_update[env] = wp.float64(0.0)
        is_outdated[env] = False


@wp.kernel
def reset_envs_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    is_outdated: wp.array(dtype=wp.bool),
    timestamp: wp.array(dtype=wp.float32),
    timestamp_last_update: wp.array(dtype=wp.float32),
    elapsed_since_update: wp.array(dtype=wp.float64),
):
    """Resets the current and last update timestamps and marks environments as outdated for those being reset.

    Args:
        reset_mask: Boolean array indicating which envs to reset.
        is_outdated: Boolean array indicating which envs need update. Will be set to True for reset envs.
        timestamp: Current timestamp per env. Will be set to 0.0 for reset envs.
        timestamp_last_update: Last update timestamp per env. Will be set to 0.0 for reset envs.
        elapsed_since_update: Time since the last refresh per env. Will be set to 0.0 for reset envs.
    """
    env = wp.tid()
    if not reset_mask[env]:
        return

    # Reset the timestamp for the sensors
    timestamp[env] = 0.0

    timestamp_last_update[env] = 0.0
    elapsed_since_update[env] = wp.float64(0.0)
    # Set all reset sensors to outdated so that they are updated when data is called the next time.
    is_outdated[env] = True
