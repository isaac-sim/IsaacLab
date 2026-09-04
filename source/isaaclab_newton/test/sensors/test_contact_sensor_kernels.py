# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton contact-sensor Warp kernels."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.sensors.contact_sensor.contact_sensor_kernels import (
    compute_first_transition_kernel,
    update_contact_sensor_kernel,
)

from isaaclab.sensors.kernels import update_outdated_envs_kernel, update_timestamp_kernel

DEVICE = "cpu"
PHYSICS_DT = 0.005
DECIMATION = 4
STEP_DT = PHYSICS_DT * DECIMATION
DWELL = 3
"""Number of policy steps the sensor stays in contact (resp. in air) before toggling."""


def _contact_schedule(num_steps: int) -> list[bool]:
    """Alternating contact/air schedule with a dwell of :attr:`DWELL` policy steps.

    Starting in contact keeps the first step unambiguous: the air timer is still zero, so only a
    first-contact event can be reported at step 0.

    Args:
        num_steps: Number of policy steps to generate.

    Returns:
        Per-step contact flags.
    """
    return [(step // DWELL) % 2 == 0 for step in range(num_steps)]


def _expected_transitions(schedule: list[bool]) -> tuple[set[int], set[int]]:
    """Reference calculation of the steps at which touchdown and lift-off occur.

    Args:
        schedule: Per-step contact flags.

    Returns:
        A tuple of the first-contact step indices and the first-air step indices.
    """
    first_contact: set[int] = set()
    first_air: set[int] = set()
    for step, in_contact in enumerate(schedule):
        # Before the first step both timers are zero, which the kernel never reports as a transition.
        was_in_contact = schedule[step - 1] if step > 0 else False
        if in_contact and not was_in_contact:
            first_contact.add(step)
        if not in_contact and was_in_contact:
            first_air.add(step)
    return first_contact, first_air


def _run_transition_probe(
    schedule: list[bool], start_clock: float, refresh_every_substep: bool, abs_tol: float
) -> tuple[set, set]:
    """Drive the update kernel over a contact schedule and collect reported transitions.

    Args:
        schedule: Per-step contact flags.
        start_clock: Initial value of the sensor clock [s]. Ageing the clock exposes the float32
            rounding error that a fixed absolute tolerance cannot absorb.
        refresh_every_substep: Whether the sensor buffers refresh on every physics substep
            (``history_length > 0``) or once per policy step (``history_length == 0``).
        abs_tol: Absolute tolerance added to the polling period to form the detection threshold [s].

    Returns:
        A tuple of the reported first-contact step indices and first-air step indices.
    """
    timestamp = wp.full(1, start_clock, dtype=wp.float32, device=DEVICE)
    timestamp_last_update = wp.full(1, start_clock, dtype=wp.float32, device=DEVICE)
    is_outdated = wp.ones(1, dtype=wp.bool, device=DEVICE)
    env_mask = wp.ones(1, dtype=wp.bool, device=DEVICE)
    forces = wp.zeros((1, 1), dtype=wp.vec3f, device=DEVICE)
    net_forces_history = wp.zeros((1, 1, 1), dtype=wp.vec3f, device=DEVICE)
    current_air_time = wp.zeros((1, 1), dtype=wp.float32, device=DEVICE)
    current_contact_time = wp.zeros((1, 1), dtype=wp.float32, device=DEVICE)
    last_air_time = wp.zeros((1, 1), dtype=wp.float32, device=DEVICE)
    last_contact_time = wp.zeros((1, 1), dtype=wp.float32, device=DEVICE)
    transition = wp.zeros((1, 1), dtype=wp.float32, device=DEVICE)

    # Matches the default ``ContactSensorCfg.update_period``: the clock is advanced on every
    # physics step, so the sensor is always flagged outdated and never skips an update.
    update_period = 0.0
    threshold = float(STEP_DT + abs_tol)

    reported_contact: set[int] = set()
    reported_air: set[int] = set()

    def _launch_update() -> None:
        wp.launch(
            update_contact_sensor_kernel,
            dim=(1, 1),
            inputs=[
                1,  # history_length
                0,  # num_filter_objects
                1.0,  # contact_force_threshold
                env_mask,
                forces,
                None,
                forces,  # net_normal_forces drives the air/contact timers
                None,
                None,
                None,
                timestamp,
                timestamp_last_update,
                net_forces_history,
                None,
                None,
                None,
                None,
                None,
                current_air_time,
                current_contact_time,
                last_air_time,
                last_contact_time,
            ],
            device=DEVICE,
        )
        wp.launch(
            update_outdated_envs_kernel,
            dim=1,
            inputs=[is_outdated, timestamp, timestamp_last_update],
            device=DEVICE,
        )

    for step, in_contact in enumerate(schedule):
        forces.assign(np.array([[[0.0, 0.0, 100.0 if in_contact else 0.0]]], dtype=np.float32))
        for _ in range(DECIMATION):
            wp.launch(
                update_timestamp_kernel,
                dim=1,
                inputs=[is_outdated, timestamp, timestamp_last_update, PHYSICS_DT, update_period],
                device=DEVICE,
            )
            if refresh_every_substep:
                _launch_update()
        if not refresh_every_substep:
            _launch_update()

        for timer, reported in ((current_contact_time, reported_contact), (current_air_time, reported_air)):
            wp.launch(
                compute_first_transition_kernel,
                dim=(1, 1),
                inputs=[threshold, timer],
                outputs=[transition],
                device=DEVICE,
            )
            if transition.numpy()[0, 0] > 0.5:
                reported.add(step)

    return reported_contact, reported_air


@pytest.mark.parametrize("refresh_every_substep", [True, False], ids=["substep_refresh", "policy_step_refresh"])
@pytest.mark.parametrize("start_clock", [0.0, 2.5, 30.0])
def test_first_transition_threshold_detects_every_transition(start_clock: float, refresh_every_substep: bool):
    """A half-update-interval tolerance flags every transition and nothing else, at any clock age.

    The sensor clock is a float32 accumulator, so after a few seconds of simulated time its rounding
    error dwarfs a fixed 1e-8 tolerance and touchdowns are silently missed (#7283).
    """
    schedule = _contact_schedule(200)
    expected_contact, expected_air = _expected_transitions(schedule)

    # Half of the sensor update interval: the midpoint between "one update ago" and "two updates ago".
    reported_contact, reported_air = _run_transition_probe(
        schedule, start_clock, refresh_every_substep, abs_tol=0.5 * PHYSICS_DT
    )

    assert reported_contact == expected_contact
    assert reported_air == expected_air
