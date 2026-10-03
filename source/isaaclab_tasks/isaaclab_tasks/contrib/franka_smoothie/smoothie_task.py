# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured task progression with scalar tap filling and no fluid simulation."""

from __future__ import annotations

import math
from enum import IntEnum

import torch

FRUIT_COUNT = 16
TARGET_VOLUME_M3 = 86.4e-6
FILL_DURATION_S = 2.0
VALVE_PRESS_M = 0.002
VALVE_RELEASE_M = 0.001


class SmoothiePhase(IntEnum):
    """Ordered physical milestones of the smoothie and blender demonstration."""

    FRUIT = 0
    TAP = 1
    LID = 2
    DOCK = 3
    BUTTON = 4
    DONE = 5


def all_fruit_delivered(whole_hull_inside: torch.Tensor) -> torch.Tensor:
    """Require every one of the 16 authored fruit hulls inside its cup.

    Args:
        whole_hull_inside: Measured whole-hull containment flags, shape [N, 16].

    Returns:
        Complete fruit delivery flags, shape [N].
    """
    if (
        not isinstance(whole_hull_inside, torch.Tensor)
        or whole_hull_inside.dtype != torch.bool
        or whole_hull_inside.ndim != 2
        or whole_hull_inside.shape[1] != FRUIT_COUNT
    ):
        raise ValueError("Require boolean whole-hull containment with shape [N, 16].")
    return whole_hull_inside.all(dim=1)


class SmoothieTaskState:
    """Track ordered task evidence independently for each environment.

    ``fill_volume_m3`` is a visual task scalar [m^3], not simulated liquid.
    ``milestones`` has shape [N, 5], ordered as fruit, tap, lid, dock and button.
    Failure latches until reset and freezes phase, filling and switch state.

    Args:
        num_envs: Number of independent environments.
        device: Torch device on which measurements and task state reside.
        step_dt_s: Elapsed simulated time per update [s].
    """

    def __init__(self, num_envs: int, device: torch.device | str, step_dt_s: float):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        if isinstance(step_dt_s, bool) or not math.isfinite(step_dt_s) or step_dt_s <= 0:
            raise ValueError("step_dt_s must be finite and positive.")
        self.num_envs = num_envs
        self.device = torch.empty(0, device=device).device
        self.step_dt_s = float(step_dt_s)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.milestones = torch.zeros((num_envs, 5), dtype=torch.bool, device=self.device)
        self.fill_volume_m3 = torch.zeros(num_envs, device=self.device)
        self.tap_on = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.tap_on_seen = torch.zeros_like(self.tap_on)
        self.tap_off_seen = torch.zeros_like(self.tap_on)
        self.failed = torch.zeros_like(self.tap_on)
        self._valve_armed = torch.zeros_like(self.tap_on)
        self._button_armed = torch.zeros_like(self.tap_on)
        self._fill_time_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)

    @property
    def fill_fraction(self) -> torch.Tensor:
        """Return the visual cup fill fraction, shape [N], bounded to [0, 1]."""
        return self.fill_volume_m3 / TARGET_VOLUME_M3

    @property
    def completed(self) -> torch.Tensor:
        """Return ordered completion without a latched failure, shape [N]."""
        return (self.phase == SmoothiePhase.DONE) & ~self.failed

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear selected environments, including valve and final-button edge history.

        Args:
            env_ids: Unique environment indices on the state device, shape [M].
                ``None`` resets every environment.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.device != self.device
            or env_ids.ndim != 1
            or bool(((env_ids < 0) | (env_ids >= self.num_envs)).any())
            or env_ids.unique().numel() != env_ids.numel()
        ):
            raise ValueError("Require unique valid environment indices on the state device.")
        for value in (
            self.phase,
            self.milestones,
            self.fill_volume_m3,
            self.tap_on,
            self.tap_on_seen,
            self.tap_off_seen,
            self.failed,
            self._valve_armed,
            self._button_armed,
            self._fill_time_s,
        ):
            value[env_ids] = 0

    @torch.no_grad()
    def update(
        self,
        *,
        fruit_delivered: torch.Tensor,
        cup_under_tap: torch.Tensor,
        cup_upright: torch.Tensor,
        cup_open: torch.Tensor,
        lid_fastened: torch.Tensor,
        cup_docked: torch.Tensor,
        blender_button_pressed: torch.Tensor,
        valve_depression_m: torch.Tensor,
        failed: torch.Tensor,
    ) -> None:
        """Consume measured evidence and advance at most one phase per update.

        Boolean measurements have shape [N]. ``fruit_delivered`` must represent
        all 16 whole fruit hulls; ``lid_fastened`` must include measured seating
        and screw engagement; ``cup_docked`` must include supported placement.
        A valve must first be released, then depressed at least 2 mm to toggle.
        It rearms only below 1 mm. Filling requires a recognized on edge during
        TAP, an open upright cup below the nozzle, and two seconds of eligible
        filling. A later off edge proves the tap milestone. A final blender
        button press is accepted only after release was observed during BUTTON.

        Args:
            fruit_delivered: All-fruit whole-hull delivery flags.
            cup_under_tap: Cup opening aligned beneath the nozzle.
            cup_upright: Cup orientation suitable for receiving liquid.
            cup_open: Cup has no lid obstructing its opening.
            lid_fastened: Measured lid seating and screw-engagement acceptance.
            cup_docked: Measured supported placement on the blender.
            blender_button_pressed: Measured mechanical blender button press.
            valve_depression_m: Measured tap button depression [m], shape [N].
            failed: Physical failure flags; true values latch until reset.
        """
        for name, value in (
            ("fruit_delivered", fruit_delivered),
            ("cup_under_tap", cup_under_tap),
            ("cup_upright", cup_upright),
            ("cup_open", cup_open),
            ("lid_fastened", lid_fastened),
            ("cup_docked", cup_docked),
            ("blender_button_pressed", blender_button_pressed),
            ("failed", failed),
        ):
            self._validate_measurement(name, value, boolean=True)
        self._validate_measurement("valve_depression_m", valve_depression_m, boolean=False)
        if not bool(torch.isfinite(valve_depression_m).all()):
            raise ValueError("Valve depression must be finite.")

        self.failed |= failed
        active = ~self.failed & (self.phase != SmoothiePhase.DONE)
        prior_phase = self.phase.clone()
        in_tap = active & (prior_phase == SmoothiePhase.TAP)
        in_button = active & (prior_phase == SmoothiePhase.BUTTON)

        valve_pressed = valve_depression_m >= VALVE_PRESS_M
        valve_released = valve_depression_m <= VALVE_RELEASE_M
        valve_edge = active & self._valve_armed & valve_pressed
        was_on = self.tap_on.clone()
        self.tap_on ^= valve_edge
        self._valve_armed[active & valve_released] = True
        self._valve_armed[active & valve_pressed] = False
        self.tap_on_seen |= in_tap & valve_edge & ~was_on

        filling = in_tap & self.tap_on & self.tap_on_seen & cup_under_tap & cup_upright & cup_open
        self._fill_time_s[filling] = (self._fill_time_s[filling] + self.step_dt_s).clamp(max=FILL_DURATION_S)
        # Remove only floating-point accumulation error at the exact fill duration.
        at_target = filling & (self._fill_time_s >= FILL_DURATION_S - 1e-12)
        self._fill_time_s[at_target] = FILL_DURATION_S
        self.fill_volume_m3[filling] = (self._fill_time_s[filling] * (TARGET_VOLUME_M3 / FILL_DURATION_S)).to(
            self.fill_volume_m3.dtype
        )
        full = self.fill_volume_m3 >= TARGET_VOLUME_M3
        self.tap_off_seen |= in_tap & valve_edge & was_on & self.tap_on_seen & full

        button_edge = in_button & self._button_armed & blender_button_pressed
        self._button_armed[in_button & ~blender_button_pressed] = True
        self._button_armed[active & blender_button_pressed] = False

        transitions = (
            active & (prior_phase == SmoothiePhase.FRUIT) & fruit_delivered,
            in_tap & self.tap_on_seen & self.tap_off_seen & ~self.tap_on & full & cup_under_tap & cup_upright,
            active & (prior_phase == SmoothiePhase.LID) & lid_fastened,
            active & (prior_phase == SmoothiePhase.DOCK) & cup_docked,
            button_edge
            & self.milestones[:, :4].all(dim=1)
            & fruit_delivered
            & full
            & ~self.tap_on
            & lid_fastened
            & cup_docked,
        )
        for phase, advance in enumerate(transitions):
            self.milestones[:, phase] |= advance
            self.phase[advance] = phase + 1

    def _validate_measurement(self, name: str, value: torch.Tensor, *, boolean: bool) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (self.num_envs,)
            or value.device != self.device
            or (value.dtype != torch.bool if boolean else not value.is_floating_point())
        ):
            kind = "boolean" if boolean else "floating"
            raise ValueError(f"{name} must be a {kind} tensor with shape [{self.num_envs}] on {self.device}.")
