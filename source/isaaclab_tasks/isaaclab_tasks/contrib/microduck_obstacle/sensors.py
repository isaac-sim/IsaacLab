# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local MicroDuck time-of-flight sensing on Newton's legacy ray-camera backend.

The native ray camera owns geometry and ideal optical-axis depths. This adapter owns
acquisition timing, episode calibration, per-frame noise and delayed packet delivery.
Policy reads consume a cached frame; they never acquire rays or resample noise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import warp as wp
from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyMultiMeshRayCasterCamera

from isaaclab.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.utils import clone, configclass
from isaaclab.utils.math import quat_apply, quat_mul


@configclass
class ToFRandomizationCfg:
    """Episode-level mounting/calibration variation and per-frame range corruption."""

    enabled: bool = True
    """Apply corruption and mounting variation; False exposes ideal native depths."""

    mount_pitch_degrees: tuple[float, float] = (0.0, 0.0)
    """Pitch about the nominal sensor left axis; positive angles point downward."""
    mount_height_m: tuple[float, float] = (0.0, 0.0)
    """Translation along the nominal sensor up axis, fixed relative to the head."""
    latency_frames: tuple[int, int] = (0, 1)
    """Inclusive episode-level transport delay in acquired frames."""
    frame_dropout_probability: float = 0.02
    """Independent packet-loss probability per acquisition; retain the last delivered frame."""
    minimum_distance: float = 0.12
    """Minimum valid optical-axis depth [m], checked before and after range noise."""
    distance_bias: tuple[float, float] = (-0.02, 0.02)
    """Episode-level additive bias shared by all zones [m]."""
    zone_bias: tuple[float, float] = (-0.01, 0.01)
    """Independent per-zone additive biases held for the episode [m]."""
    distance_scale: tuple[float, float] = (0.98, 1.02)
    """Episode-level multiplicative depth calibration [unitless]."""
    noise_std: tuple[float, float] = (0.003, 0.01)
    """Episode-level base standard deviation for independent per-acquisition noise [m]."""
    distance_noise_fraction: float = 0.008
    """Additional noise standard deviation as a fraction of ideal depth."""
    invalid_zone_probability: float = 0.03
    """Independent non-floor dropout per zone/acquisition; an uncalibrated generic prior.

    Mixed-scene row validity percentages also include missing floor returns, so they
    must not be used as conditional dropout probabilities for walls and obstacles.
    """
    near_floor_edge_impairment_probability: float = 0.0
    """Episode probability of attenuating one uniformly selected outer column.

    This mirrored stress augmentation covers observed near-floor edge loss; the
    episode mixture and persistence are not fitted hardware frequencies or timing.
    """
    near_floor_edge_survival_probability: tuple[float, float] = (0.02, 0.55)
    """Episode survival probability for independent near-floor edge returns per acquisition."""
    floor_validity_midpoint_offset_m: tuple[float, float] = (0.0, 0.0)
    """Episode-level shift of the far-floor validity midpoint; a sensor/material prior."""
    floor_validity_transition_m: tuple[float, float] | None = (0.556, 0.015)
    """Flat-floor validity sigmoid (midpoint, width) in axial metres; None disables it.

    Near-floor hits through 0.55 m remain valid except for optional edge stress.
    Farther hits use a provisional sigmoid with an episode-level midpoint shift.
    With zero shift it is 16% valid at 0.58 m and almost never beyond 0.65 m.
    Floor returns replace the generic prior, preserving reliable near-floor observations.
    This is a distance proxy for grazing-incidence loss at the measured standing pose.
    """


class MicroDuckToFCamera(LegacyMultiMeshRayCasterCamera):
    """Use Newton's legacy multi-mesh ray camera to deliver coherent, optionally delayed frames.

    Deadlines retain sub-tick remainder, giving 15 Hz on average. Captures are
    quantized to scene updates (20 ms with Newton-owned decimation). Episode
    resets clear only the selected rows of frame history and
    redraw their calibration/noise parameters. The shared acquisition phase stays fixed.
    """

    cfg: MicroDuckToFCameraCfg
    """Task sensor configuration; ``sample_period`` retains the requested acquisition interval."""

    output_name = "distance_to_image_plane"

    def __init__(self, cfg: MicroDuckToFCameraCfg) -> None:
        self.sample_period = cfg.update_period
        if not math.isfinite(self.sample_period) or self.sample_period <= 0:
            raise ValueError("Head sensor update_period must be positive and finite.")
        lo, hi = cfg.randomization.latency_frames
        if not 0 <= lo <= hi:
            raise ValueError("Sensor latency_frames must be an ordered nonnegative interval.")
        for name in ("mount_pitch_degrees", "mount_height_m", "floor_validity_midpoint_offset_m"):
            low, high = getattr(cfg.randomization, name)
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError(f"Sensor {name} must be a finite ordered interval.")
        if not 0.0 <= cfg.randomization.invalid_zone_probability <= 1.0:
            raise ValueError("Zone dropout must be a probability.")
        if not 0.0 <= cfg.randomization.near_floor_edge_impairment_probability <= 1.0:
            raise ValueError("Near-floor edge impairment must be a probability.")
        low, high = cfg.randomization.near_floor_edge_survival_probability
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("Near-floor edge survival must be an ordered probability interval.")
        transition = cfg.randomization.floor_validity_transition_m
        if transition is not None and (len(transition) != 2 or not all(math.isfinite(v) and v > 0 for v in transition)):
            raise ValueError("Floor validity requires a positive finite midpoint and width.")
        self._acquisition_time = 0.0
        self._next_capture = self.sample_period
        # Only the deadline scheduler triggers periodic captures; reset captures remain native.
        native_cfg = clone(cfg)
        native_cfg.update_period = float("inf")
        super().__init__(native_cfg)

    @property
    def supports_batch_update(self) -> bool:
        # Scene updates must use this sensor's acquisition and delivery schedule.
        return False

    @property
    def sample_time(self) -> torch.Tensor:
        """Latest native acquisition time [s], shape (num_envs,)."""
        return wp.to_torch(self._timestamp_last_update)

    @property
    def delivered_sample_time(self) -> torch.Tensor:
        """Delivered frame acquisition time [s], shape (num_envs,); -1 before first delivery."""
        return self._delivered_timestamp if self.cfg.randomization.enabled else self.sample_time

    @property
    def sample_age(self) -> torch.Tensor:
        """Delivered frame age [s], shape (num_envs,), including latency and packet loss."""
        return wp.to_torch(self._timestamp) - self.delivered_sample_time

    @property
    def observation(self) -> torch.Tensor:
        """Cached optical-axis depths [m], shape (num_envs, height, width, 1).

        Zero denotes an invalid return. Reading this tensor does not advance the
        sensor or resample noise. Callers must not modify the returned buffer.
        """
        return self._delivered_frame if self.cfg.randomization.enabled else self._data.output[self.output_name].torch

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Clear selected packet histories and resample their episode calibration."""
        reset_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(env_mask=reset_mask)
        ids = wp.to_torch(reset_mask).nonzero(as_tuple=False).flatten()
        self._frame_buffer[:, ids] = 0
        self._frame_timestamps[:, ids] = -1.0
        self._write_indices[ids] = 0
        self._delivered_frame[ids] = 0
        self._delivered_timestamp[ids] = -1.0
        self._reset_randomization(ids)

    def update(self, dt: float, force_recompute: bool = False) -> None:
        """Advance the acquisition clock without allowing forced reads to bypass the sensor rate."""
        super().update(dt, force_recompute=False)
        if not self.is_initialized:
            return
        self._acquisition_time += dt
        if self._acquisition_time + 1e-9 >= self._next_capture:
            periods = math.floor((self._acquisition_time + 1e-9 - self._next_capture) / self.sample_period) + 1
            self._next_capture += periods * self.sample_period
            self._invalidate_buffers()
            self._update_outdated_buffers()

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self._nominal_offset_pos = wp.to_torch(self._offset_pos_wp).clone()
        self._nominal_offset_quat = wp.to_torch(self._offset_quat_wp).clone()
        raw = self._data.output[self.output_name].torch
        slots = self.cfg.randomization.latency_frames[1] + 1
        # Ring-buffer axes: (history slot, environment, image row, image column, channel).
        self._zone_columns = torch.arange(raw.shape[2], device=raw.device)[None, None, :, None]
        self._frame_buffer = torch.zeros((slots, *raw.shape), dtype=raw.dtype, device=raw.device)
        self._frame_timestamps = torch.full((slots, self._num_envs), -1.0, device=raw.device)
        self._delivered_frame = torch.zeros_like(raw)
        self._delivered_timestamp = torch.full((self._num_envs,), -1.0, device=raw.device)
        self._write_indices = torch.zeros(self._num_envs, dtype=torch.long, device=raw.device)
        self._latency_frames = torch.zeros(self._num_envs, dtype=torch.long, device=raw.device)
        self._near_floor_edge_column = torch.full((self._num_envs, 1, 1, 1), -1, dtype=torch.long, device=raw.device)
        self._episode_parameters = {
            name: torch.empty((self._num_envs, 1, 1, 1), device=raw.device)
            for name in (
                "distance_bias",
                "distance_scale",
                "noise_std",
                "floor_validity_midpoint_offset_m",
                "near_floor_edge_survival_probability",
            )
        }
        self._episode_parameters["zone_bias"] = torch.empty_like(raw)
        self._reset_randomization(torch.arange(self._num_envs, device=raw.device))

    def _reset_randomization(self, env_ids: torch.Tensor) -> None:
        """Sample calibration and articulated mount offsets for the selected environments."""
        low, high = self.cfg.randomization.latency_frames
        self._latency_frames[env_ids] = torch.randint(
            low, high + 1, (len(env_ids),), device=self._latency_frames.device
        )
        for name, values in self._episode_parameters.items():
            low, high = getattr(self.cfg.randomization, name)
            values[env_ids] = torch.empty_like(values[env_ids]).uniform_(low, high)

        edge_shape = (len(env_ids), 1, 1, 1)
        impaired = (
            torch.rand(edge_shape, device=self._device) < self.cfg.randomization.near_floor_edge_impairment_probability
        )
        edge = torch.randint(0, 2, edge_shape, device=self._device) * (self._delivered_frame.shape[2] - 1)
        self._near_floor_edge_column[env_ids] = torch.where(impaired, edge, -1)

        self._reset_mount(env_ids)

    def _reset_mount(self, env_ids: torch.Tensor) -> None:
        """Sample the selected sensors' offsets relative to the articulated head."""
        nominal_pos = self._nominal_offset_pos[env_ids]
        nominal_quat = self._nominal_offset_quat[env_ids]
        if not self.cfg.randomization.enabled:
            wp.to_torch(self._offset_pos_wp)[env_ids] = nominal_pos
            wp.to_torch(self._offset_quat_wp)[env_ids] = nominal_quat
            return
        height = torch.zeros_like(nominal_pos)
        height[:, 2].uniform_(*self.cfg.randomization.mount_height_m)
        pitch = torch.empty(len(env_ids), device=self._device).uniform_(*self.cfg.randomization.mount_pitch_degrees)
        pitch *= math.pi / 180.0
        rotation = torch.zeros_like(nominal_quat)
        rotation[:, 1] = torch.sin(pitch / 2.0)
        rotation[:, 3] = torch.cos(pitch / 2.0)
        # Update native mounting buffers so every ray follows the articulated head.
        wp.to_torch(self._offset_pos_wp)[env_ids] = nominal_pos + quat_apply(nominal_quat, height)
        wp.to_torch(self._offset_quat_wp)[env_ids] = quat_mul(nominal_quat, rotation)

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        """Acquire ideal depths, corrupt one packet, then deliver the delayed packet."""
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        super()._update_buffers_impl(env_mask)
        if not self.cfg.randomization.enabled:
            return
        ideal_depths = self._data.output[self.output_name].torch[env_ids]
        write_slots = self._write_indices[env_ids]
        self._frame_buffer[write_slots, env_ids] = self._corrupt_frame(ideal_depths, env_ids)
        packet_received = (
            torch.rand(len(env_ids), device=ideal_depths.device) >= self.cfg.randomization.frame_dropout_probability
        )
        capture_time = wp.to_torch(self._timestamp)[env_ids]
        self._frame_timestamps[write_slots, env_ids] = torch.where(packet_received, capture_time, -1.0)
        read_slots = (write_slots - self._latency_frames[env_ids]) % len(self._frame_buffer)
        packet_times = self._frame_timestamps[read_slots, env_ids]
        # Dropped or not-yet-available packets retain the previous frame and its true age.
        packet_available = packet_times >= 0.0
        delivery_ids = env_ids[packet_available]
        self._delivered_frame[delivery_ids] = self._frame_buffer[read_slots[packet_available], delivery_ids]
        self._delivered_timestamp[delivery_ids] = packet_times[packet_available]
        self._write_indices[env_ids] = (write_slots + 1) % len(self._frame_buffer)

    def _corrupt_frame(self, ideal_depths: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Apply range noise and zone validity to one acquisition; invalid depths become zero."""
        cfg = self.cfg.randomization
        calibration = self._episode_parameters
        valid_depth = (
            torch.isfinite(ideal_depths)
            & (ideal_depths >= cfg.minimum_distance)
            & (ideal_depths <= self.cfg.max_distance)
        )
        noisy_depths = ideal_depths * calibration["distance_scale"][env_ids] + calibration["distance_bias"][env_ids]
        noisy_depths = noisy_depths + calibration["zone_bias"][env_ids]
        noise_std = calibration["noise_std"][env_ids] + cfg.distance_noise_fraction * ideal_depths
        noisy_depths = noisy_depths + torch.randn_like(ideal_depths) * noise_std
        dropout_probability = self._zone_dropout_probability(ideal_depths, env_ids)
        valid_depth &= torch.rand_like(ideal_depths) >= dropout_probability
        valid_depth &= (noisy_depths >= cfg.minimum_distance) & (noisy_depths <= self.cfg.max_distance)
        return torch.where(valid_depth, noisy_depths, 0.0)

    def _zone_dropout_probability(self, ideal_depths: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Combine the non-floor, grazing-floor and near-floor edge dropout priors."""
        cfg = self.cfg.randomization
        calibration = self._episode_parameters
        dropout_probability = cfg.invalid_zone_probability
        # This task's flat floor is at world z=0. Wall/obstacle hits keep the generic prior.
        floor_hit = self.ray_hits_w.torch[env_ids, :, 2].reshape_as(ideal_depths).abs() < 1e-3
        if cfg.floor_validity_transition_m is not None:
            midpoint, width = cfg.floor_validity_transition_m
            midpoint = midpoint + calibration["floor_validity_midpoint_offset_m"][env_ids]
            floor_dropout = torch.sigmoid((ideal_depths - midpoint) / width)
            floor_dropout = torch.where(ideal_depths <= 0.55, 0.0, floor_dropout)
            dropout_probability = torch.where(floor_hit, floor_dropout, dropout_probability)
        impaired_column = self._zone_columns == self._near_floor_edge_column[env_ids]
        near_floor_edge = floor_hit & (ideal_depths >= cfg.minimum_distance) & (ideal_depths <= 0.55) & impaired_column
        edge_dropout = 1.0 - (1.0 - dropout_probability) * calibration["near_floor_edge_survival_probability"][env_ids]
        dropout_probability = torch.where(near_floor_edge, edge_dropout, dropout_probability)
        return dropout_probability


@configclass
class MicroDuckToFCameraCfg(MultiMeshRayCasterCameraCfg):
    """Task-local ToF sensor with an explicit ideal-sensor evaluation switch."""

    class_type: type = MicroDuckToFCamera
    randomization: ToFRandomizationCfg = ToFRandomizationCfg()
