# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-level regression tests for contact-sensor first-contact/first-air detection.

The contact-sensor kernels are exercised directly with a synthetic contact
signal -- no physics, no stage, no AppLauncher. The sensor is in contact for
one policy step and in the air for the next, with every transition landing
exactly on a policy-step boundary, so *every* touchdown and lift-off must be
reported by ``compute_first_contact`` / ``compute_first_air``.

Both sensor update cadences are covered:

* eager (``history_length > 0``): buffers refresh every physics substep
* lazy (``history_length == 0``): buffers refresh once per policy step, so
  ``current_*_time`` at a transition sample equals exactly one sensor update
  interval and the ``t < dt + abs_tol`` comparison holds only by the
  tolerance margin.

Because the float32 timestamps grow with the simulation clock, the timer
rounding error scales with the clock magnitude, and a fixed ``abs_tol=1e-8``
silently misses most transitions (issue #7283). These tests drive the clock
through the worst-case bands reported there, and beyond.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import warp as wp

# ---------------------------------------------------------------------------
# Load kernel modules directly (avoids Isaac Sim / Omniverse dependencies)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir))


def _load_module(name: str, rel_path: str):
    path = os.path.join(_REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core_mod = _load_module("core_sensor_kernels", os.path.join("source", "isaaclab", "isaaclab", "sensors", "kernels.py"))
_physx_mod = _load_module(
    "physx_contact_kernels",
    os.path.join("source", "isaaclab_physx", "isaaclab_physx", "sensors", "contact_sensor", "kernels.py"),
)
_newton_mod = _load_module(
    "newton_contact_kernels",
    os.path.join(
        "source", "isaaclab_newton", "isaaclab_newton", "sensors", "contact_sensor", "contact_sensor_kernels.py"
    ),
)
_ovphysx_mod = _load_module(
    "ovphysx_contact_kernels",
    os.path.join("source", "isaaclab_ovphysx", "isaaclab_ovphysx", "sensors", "contact_sensor", "kernels.py"),
)


def _kernel_arg_labels(kernel) -> list[str]:
    """Names of a warp kernel's parameters (warp stores them on the adjoint)."""
    return [str(arg.label) for arg in kernel.adj.args]


# ---------------------------------------------------------------------------
# Synthetic alternating-contact rollout shared by all backends
# ---------------------------------------------------------------------------

_DEVICE = "cpu"
_PHYSICS_DT = 0.005
_DECIMATION = 4
_STEP_DT = _PHYSICS_DT * _DECIMATION
_NUM_ENVS = 1
_NUM_SENSORS = 1


class _ContactRollout:
    """Drives one backend's kernels through an alternating contact signal."""

    def __init__(self, update_fn, first_contact_fn, update_period: float):
        self._update_fn = update_fn
        self._first_contact_fn = first_contact_fn
        self._update_period = update_period

        self.timestamp = wp.zeros(_NUM_ENVS, dtype=wp.float32, device=_DEVICE)
        self.timestamp_last = wp.zeros(_NUM_ENVS, dtype=wp.float32, device=_DEVICE)
        self.is_outdated = wp.ones(_NUM_ENVS, dtype=wp.bool, device=_DEVICE)
        self.env_mask = wp.ones(_NUM_ENVS, dtype=wp.bool, device=_DEVICE)
        self.forces = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.vec3f, device=_DEVICE)
        self.current_air = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        self.current_contact = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        self.last_air = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        self.last_contact = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        self.transition = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        # Transition latches and first-phase timers added by the fix for #7283.
        # Both stay unwritten when the tree is unfixed, which is how the
        # regression tests below detect the missing fix.
        self.first_contact_latch = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        self.first_air_latch = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        self.first_contact_time = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        self.first_air_time = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.float32, device=_DEVICE)
        # PhysX/ovphysx kernels expect (N*B,) flat force buffers.
        self.forces_flat = wp.zeros(_NUM_ENVS * _NUM_SENSORS, dtype=wp.vec3f, device=_DEVICE)
        # Newton's update kernel takes a (possibly None) history buffer.
        self.history = wp.zeros((_NUM_ENVS, 1, _NUM_SENSORS), dtype=wp.vec3f, device=_DEVICE)
        self.net_forces_w = wp.zeros((_NUM_ENVS, _NUM_SENSORS), dtype=wp.vec3f, device=_DEVICE)
        self.net_forces_w_history = wp.zeros((_NUM_ENVS, 1, _NUM_SENSORS), dtype=wp.vec3f, device=_DEVICE)

    # -- launch helpers ------------------------------------------------------

    def _step_timestamp(self):
        wp.launch(
            _core_mod.update_timestamp_kernel,
            dim=_NUM_ENVS,
            inputs=[
                self.is_outdated,
                self.timestamp,
                self.timestamp_last,
                _PHYSICS_DT,
                float(self._update_period),
            ],
            device=_DEVICE,
        )

    def _step_update(self, in_contact: bool):
        force = np.array([100.0, 0.0, 0.0] if in_contact else [0.0, 0.0, 0.0], dtype=np.float32)
        self.forces.assign(force.reshape(_NUM_ENVS, _NUM_SENSORS, 3))
        self.forces_flat.assign(force.reshape(_NUM_ENVS * _NUM_SENSORS, 3))
        self._update_fn(self)

    def _is_outdated(self) -> bool:
        return bool(self.is_outdated.numpy()[0])

    def _step_outdated(self):
        wp.launch(
            _core_mod.update_outdated_envs_kernel,
            dim=_NUM_ENVS,
            inputs=[self.is_outdated, self.timestamp, self.timestamp_last],
            device=_DEVICE,
        )

    def _reset_timestamps(self):
        wp.launch(
            _core_mod.reset_envs_kernel,
            dim=_NUM_ENVS,
            inputs=[self.env_mask, self.is_outdated, self.timestamp, self.timestamp_last],
            device=_DEVICE,
        )

    def _reset_buffers(self):
        """Mimics the sensor's reset(): zero the air/contact timers and latches."""
        for buf in (
            self.current_air,
            self.current_contact,
            self.last_air,
            self.last_contact,
            self.first_contact_latch,
            self.first_air_latch,
            self.first_contact_time,
            self.first_air_time,
        ):
            buf.zero_()

    # -- driver --------------------------------------------------------------

    def _substep(self, in_contact: bool) -> bool:
        """One physics substep; returns True if the sensor buffers were refreshed."""
        self._step_timestamp()
        force = np.array([100.0, 0.0, 0.0] if in_contact else [0.0, 0.0, 0.0], dtype=np.float32)
        self.forces.assign(force.reshape(_NUM_ENVS, _NUM_SENSORS, 3))
        self.forces_flat.assign(force.reshape(_NUM_ENVS * _NUM_SENSORS, 3))
        if self._is_outdated():
            self._update_fn(self)
            self._step_outdated()
            return True
        return False

    def run(self, n_steps: int, abs_tol: float = 1.0e-8) -> tuple[int, int]:
        """Runs an alternating-contact rollout; returns (missed touchdowns, missed lift-offs).

        Each contact phase lasts one policy step and every transition lands on
        a buffer-refresh boundary, so the accumulated timer at the transition
        sample equals exactly one policy step -- the worst case from the issue,
        where the ``t < dt + abs_tol`` comparison holds only by the tolerance
        margin and float32 clock rounding tips it over.
        """
        self._reset_timestamps()
        # One air phase first, so the first contact phase is a genuine touchdown.
        self._phase(False)
        missed_td = 0
        missed_lo = 0
        for step in range(n_steps):
            in_contact = step % 2 == 0
            self._phase(in_contact)
            first = self._first_contact_fn(self, _STEP_DT, abs_tol, in_contact)
            if first == 0.0:
                if in_contact:
                    missed_td += 1
                else:
                    missed_lo += 1
        return missed_td, missed_lo

    def _phase(self, in_contact: bool) -> None:
        """Advances one policy-length contact phase with the transition on a refresh boundary."""
        if self._update_period <= _PHYSICS_DT + 1.0e-9:
            # Eager cadence: every substep refreshes the buffers.
            for _ in range(_DECIMATION):
                self._substep(in_contact)
        else:
            # Lazy cadence: keep substepping until the next refresh consumes
            # the new phase (guards against the float32 update-period race).
            refreshed = False
            guard = 0
            while not refreshed:
                refreshed = self._substep(in_contact)
                guard += 1
                assert guard < 3 * _DECIMATION, "sensor never refreshed within a policy step"


# ---------------------------------------------------------------------------
# Per-backend wiring
# ---------------------------------------------------------------------------


def _update_newton(rec: _ContactRollout):
    inputs = [
        1,  # history_length (unused when the history buffer is None)
        1.0,  # contact_force_threshold
        rec.env_mask,
        rec.forces,
        rec.timestamp,
        rec.timestamp_last,
    ]
    outputs = [
        None,  # net_forces_history (optional)
        rec.current_air,
        rec.current_contact,
        rec.last_air,
        rec.last_contact,
    ]
    if "first_contact_latch" in _kernel_arg_labels(_newton_mod.update_contact_sensor_kernel):
        outputs += [rec.first_contact_latch, rec.first_air_latch, rec.first_contact_time, rec.first_air_time]
    wp.launch(
        _newton_mod.update_contact_sensor_kernel,
        dim=(_NUM_ENVS, _NUM_SENSORS),
        inputs=inputs,
        outputs=outputs,
        device=_DEVICE,
    )


def _update_physx(rec: _ContactRollout):
    inputs = [
        rec.forces_flat,
        None,  # net_forces_matrix_flat (optional)
        rec.env_mask,
        _NUM_SENSORS,
        0,  # num_filter_shapes
        1,  # history_length
        1.0,  # contact_force_threshold
        rec.timestamp,
        rec.timestamp_last,
    ]
    outputs = [
        rec.net_forces_w,
        rec.net_forces_w_history,
        None,  # force_matrix_w (optional)
        None,  # force_matrix_w_history (optional)
        rec.current_air,
        rec.current_contact,
        rec.last_air,
        rec.last_contact,
    ]
    if "first_contact_latch" in _kernel_arg_labels(_physx_mod.update_net_forces_kernel):
        outputs += [rec.first_contact_latch, rec.first_air_latch, rec.first_contact_time, rec.first_air_time]
    wp.launch(
        _physx_mod.update_net_forces_kernel,
        dim=(_NUM_ENVS, _NUM_SENSORS),
        inputs=inputs,
        outputs=outputs,
        device=_DEVICE,
    )


def _update_ovphysx(rec: _ContactRollout):
    inputs = [
        rec.forces_flat,
        None,  # net_forces_matrix_flat (optional)
        rec.env_mask,
        _NUM_ENVS,
        _NUM_SENSORS,
        0,  # num_filter_shapes
        1,  # history_length
        1.0,  # contact_force_threshold
        rec.timestamp,
        rec.timestamp_last,
    ]
    outputs = [
        rec.net_forces_w,
        rec.net_forces_w_history,
        None,  # force_matrix_w (optional)
        None,  # force_matrix_w_history (optional)
        rec.current_air,
        rec.current_contact,
        rec.last_air,
        rec.last_contact,
    ]
    if "first_contact_latch" in _kernel_arg_labels(_ovphysx_mod.update_net_forces_ovphysx_kernel):
        outputs += [rec.first_contact_latch, rec.first_air_latch, rec.first_contact_time, rec.first_air_time]
    wp.launch(
        _ovphysx_mod.update_net_forces_ovphysx_kernel,
        dim=(_NUM_ENVS, _NUM_SENSORS),
        inputs=inputs,
        outputs=outputs,
        device=_DEVICE,
    )


def _compute_first_newton(rec: _ContactRollout, dt: float, abs_tol: float, want_contact: bool = True) -> float:
    kernel = getattr(_newton_mod, "compute_first_transition_kernel", None)
    if kernel is not None and "transition_latch" in _kernel_arg_labels(kernel):
        latch = rec.first_contact_latch if want_contact else rec.first_air_latch
        latch_time = rec.first_contact_time if want_contact else rec.first_air_time
        wp.launch(
            kernel,
            dim=(_NUM_ENVS, _NUM_SENSORS),
            inputs=[float(dt), float(abs_tol), latch, latch_time],
            outputs=[rec.transition],
            device=_DEVICE,
        )
    else:
        wp.launch(
            _newton_mod.compute_first_transition_kernel,
            dim=(_NUM_ENVS, _NUM_SENSORS),
            inputs=[float(dt + abs_tol), rec.current_contact if want_contact else rec.current_air],
            outputs=[rec.transition],
            device=_DEVICE,
        )
    return float(rec.transition.numpy()[0, 0])


def _compute_first_physx(rec: _ContactRollout, dt: float, abs_tol: float, want_contact: bool = True) -> float:
    kernel = getattr(_physx_mod, "compute_first_transition_kernel", None)
    if kernel is not None and "transition_latch" in _kernel_arg_labels(kernel):
        latch = rec.first_contact_latch if want_contact else rec.first_air_latch
        latch_time = rec.first_contact_time if want_contact else rec.first_air_time
        wp.launch(
            kernel,
            dim=(_NUM_ENVS, _NUM_SENSORS),
            inputs=[float(dt), float(abs_tol), latch, latch_time],
            outputs=[rec.transition],
            device=_DEVICE,
        )
    else:
        wp.launch(
            _physx_mod.compute_first_transition_kernel,
            dim=(_NUM_ENVS, _NUM_SENSORS),
            inputs=[float(dt + abs_tol), rec.current_contact if want_contact else rec.current_air],
            outputs=[rec.transition],
            device=_DEVICE,
        )
    return float(rec.transition.numpy()[0, 0])


def _compute_first_ovphysx(rec: _ContactRollout, dt: float, abs_tol: float, want_contact: bool = True) -> float:
    kernel = getattr(_ovphysx_mod, "compute_first_transition_kernel", None)
    if kernel is not None and "transition_latch" in _kernel_arg_labels(kernel):
        latch = rec.first_contact_latch if want_contact else rec.first_air_latch
        latch_time = rec.first_contact_time if want_contact else rec.first_air_time
        wp.launch(
            kernel,
            dim=(_NUM_ENVS, _NUM_SENSORS),
            inputs=[float(dt), float(abs_tol), latch, latch_time],
            outputs=[rec.transition],
            device=_DEVICE,
        )
    else:
        wp.launch(
            _ovphysx_mod.compute_first_transition_kernel,
            dim=(_NUM_ENVS, _NUM_SENSORS),
            inputs=[float(dt + abs_tol), rec.current_contact if want_contact else rec.current_air],
            outputs=[rec.transition],
            device=_DEVICE,
        )
    return float(rec.transition.numpy()[0, 0])


# A backend case: (name, rollout factory kwargs).
# update_period=0.0 -> is_outdated stays True -> buffers refresh every substep
# after the first policy step (eager cadence, i.e. history_length > 0).
# update_period=step_dt -> buffers refresh once per policy step (lazy
# cadence, i.e. history_length == 0).
_BACKENDS = {
    "newton": lambda period: _ContactRollout(_update_newton, _compute_first_newton, period),
    "physx": lambda period: _ContactRollout(_update_physx, _compute_first_physx, period),
    "ovphysx": lambda period: _ContactRollout(_update_ovphysx, _compute_first_ovphysx, period),
}
_CADENCES = {
    "eager": _PHYSICS_DT,  # buffers refreshed every physics substep
    "lazy": _STEP_DT,  # buffers refreshed once per policy step
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["newton", "physx", "ovphysx"])
@pytest.mark.parametrize("cadence", ["eager", "lazy"])
class TestFirstContactDetection:
    """Every boundary-aligned transition must be reported, regardless of clock."""

    def test_short_rollout(self, backend, cadence):
        """20 s alternating contact: the exact reproducer from the issue."""
        rollout = _BACKENDS[backend](_CADENCES[cadence])
        missed_td, missed_lo = rollout.run(1000)
        assert missed_td == 0, f"{backend}/{cadence}: missed {missed_td}/500 touchdowns"
        assert missed_lo == 0, f"{backend}/{cadence}: missed {missed_lo}/500 lift-offs"

    def test_aged_clock(self, backend, cadence):
        """60 s rollout: the clock passes the ULP-4e-6 band around 2-16 s and the
        1e-5 ULP band past 16 s; detection must not depend on the clock magnitude."""
        rollout = _BACKENDS[backend](_CADENCES[cadence])
        missed_td, missed_lo = rollout.run(3000)
        assert missed_td == 0, f"{backend}/{cadence}: missed {missed_td}/1500 touchdowns"
        assert missed_lo == 0, f"{backend}/{cadence}: missed {missed_lo}/1500 lift-offs"


@pytest.mark.parametrize("backend", ["newton", "physx", "ovphysx"])
def test_latch_persistence_eager(backend):
    """With substep updates, a transition missed by the lazy refetch must persist
    until the policy step reads it (the latch must not be overwritten by later
    substeps inside the same policy step)."""
    rollout = _BACKENDS[backend](_CADENCES["eager"])
    rollout._reset_timestamps()
    while not rollout._substep(False):  # start in the air (one refresh)
        pass
    # Two policy steps: in contact, then in the air. With eager cadence each
    # substep is a refresh, so the touchdown refresh is followed by three more
    # refreshes inside the same policy step before the query below.
    for step, in_contact in enumerate([True, False]):
        for _ in range(_DECIMATION):
            rollout._substep(in_contact)
        if step == 0:
            # Touchdown happened inside the first policy step; the latch must
            # still be set when the next policy step queries it.
            first = rollout._first_contact_fn(rollout, _STEP_DT, 1.0e-8)
            assert first == 1.0, f"{backend}: touchdown latch did not persist across the policy step"
        else:
            first = rollout._first_contact_fn(rollout, _STEP_DT, 1.0e-8)
            assert first == 0.0, f"{backend}: stale touchdown latch after the contact phase ended"


@pytest.mark.parametrize("backend", ["newton", "physx", "ovphysx"])
def test_first_phase_after_reset_is_reported(backend):
    """A body that starts an episode already in contact (or in air) must be
    reported on the first refresh after reset, matching the pre-latch timer
    behaviour (see the review of #7294)."""
    rollout = _BACKENDS[backend](_CADENCES["lazy"])
    rollout._reset_timestamps()
    rollout._reset_buffers()
    # Age the clock first so the refresh after reset happens at a large sim time.
    for _ in range(200):
        rollout._substep(True)
    rollout._reset_timestamps()
    rollout._reset_buffers()
    for in_contact, want in [(True, 1.0), (False, 1.0)]:
        rollout._reset_timestamps()
        rollout._reset_buffers()
        refreshed = False
        while not refreshed:
            refreshed = rollout._substep(in_contact)
        got = rollout._first_contact_fn(rollout, _STEP_DT, 1.0e-8, in_contact)
        assert got == want, f"{backend}: initial phase (in_contact={in_contact}) not reported after reset"


@pytest.mark.parametrize("backend", ["newton", "physx", "ovphysx"])
def test_no_false_positive_when_contact_steady(backend):
    """A sensor that has been in contact for longer than dt must not be reported
    as a first contact."""
    rollout = _BACKENDS[backend](_CADENCES["lazy"])
    rollout.run(2)  # settle alternating state (ends in the air)
    # The first contact refresh after an air phase is a genuine touchdown; consume it.
    refreshed = False
    while not refreshed:
        refreshed = rollout._substep(True)
    assert rollout._first_contact_fn(rollout, _STEP_DT, 1.0e-8) == 1.0, f"{backend}: touchdown not reported"
    for _ in range(10):  # ten further refresh intervals of continuous contact
        refreshed = False
        while not refreshed:
            refreshed = rollout._substep(True)
        first = rollout._first_contact_fn(rollout, _STEP_DT, 1.0e-8)
        assert first == 0.0, f"{backend}: false first-contact during steady contact"
