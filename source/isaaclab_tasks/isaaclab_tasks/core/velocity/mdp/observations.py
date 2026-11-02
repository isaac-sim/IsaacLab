# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Noise-free physical observations used by the differentiable velocity task."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase


def _feet(env) -> tuple[list[int], list[int]]:
    """Resolve the reward's feet in articulation and contact-sensor order."""
    sensor_cfg = env.cfg.rewards.feet_air_time.params["sensor_cfg"]
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_ids, names = sensor.find_sensors(sensor_cfg.body_names, preserve_order=True)
    asset_ids, _ = env.scene["robot"].find_bodies(names, preserve_order=True)
    return asset_ids, sensor_ids


class VelocityState(ManagerTermBase):
    """Read physical state without modifying observation history or contact timers.

    Order: body linear/angular velocity, root XYZW quaternion, height, joint
    position/velocity, foot world velocity, current air/contact and last air time.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._feet, self._sensor_feet = _feet(env)

    def __call__(self, env) -> torch.Tensor:
        data = env.scene["robot"].data
        contact = env.scene.sensors[env.cfg.rewards.feet_air_time.params["sensor_cfg"].name].data
        return torch.cat(
            (
                data.root_lin_vel_b.torch,
                data.root_ang_vel_b.torch,
                data.root_quat_w.torch,
                data.root_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3],
                data.joint_pos.torch,
                data.joint_vel.torch,
                data.body_lin_vel_w.torch[:, self._feet].flatten(1),
                contact.current_air_time.torch[:, self._sensor_feet],
                contact.current_contact_time.torch[:, self._sensor_feet],
                contact.last_air_time.torch[:, self._sensor_feet],
            ),
            dim=-1,
        )


class VelocityLabels(ManagerTermBase):
    """Current foot contacts, all-body force-history contacts, first foot contacts.

    Timer contacts use the sensor's own threshold through its contact clock.
    Force-history contacts use 1 N, matching velocity sliding and illegal contact.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        _, self._feet = _feet(env)

    def __call__(self, env) -> torch.Tensor:
        sensor = env.scene.sensors[env.cfg.rewards.feet_air_time.params["sensor_cfg"].name]
        forces = sensor.data.net_normal_forces_w_history.torch
        return torch.cat(
            (
                sensor.data.current_contact_time.torch[:, self._feet] > 0,
                forces.norm(dim=-1).amax(dim=1) > 1.0,
                sensor.compute_first_contact(env.step_dt).torch[:, self._feet],
            ),
            dim=-1,
        ).float()


def velocity_context(env) -> torch.Tensor:
    """Read privileged body masses for world-model conditioning and the critic."""
    return env.scene["robot"].data.body_mass.torch
