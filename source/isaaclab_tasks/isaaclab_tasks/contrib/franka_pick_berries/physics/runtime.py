# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live MPM contact and reciprocal finger force at the robot's physics rate."""

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from ..assets.usd_asset import load_berry
from ..scene.tableware import BOWL
from .contact import pad_contact
from .materials import simulation_parameters, workcell_grid
from .mpm.explicit_mpm import ExplicitMPM
from .mpm.hand_contact import HandContact, interpolate_pose, particle_contact, sticky_contact
from .mpm.state_check import StateCheck
from .pair import combine_tissue, random_poses
from .resolution import physics_resolution
from .tableware_contact import TablewareContact


@wp.kernel
def adhesive_impulse(
    attached: wp.array[int],
    force: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    dt: float,
):
    p = wp.tid()
    if attached[p] >= 0:
        wp.atomic_add(impulse, attached[p], dt * force[p])


@wp.kernel
def live_colliders(
    ticks: wp.array[int],
    start: wp.array[int],
    dt: float,
    samples: wp.array2d[wp.transform],
    poses: wp.array[wp.transform],
    following: wp.array[wp.transform],
):
    i = wp.tid()
    t = float(ticks[0] - start[0] - 1) * dt * 120.0
    poses[i] = interpolate_pose(samples, i, wp.clamp(t, 0.0, 1.0))
    following[i] = interpolate_pose(samples, i, wp.clamp(t + dt * 120.0, 0.0, 1.0))


class LiveContact(HandContact):
    """Interpolate measured finger poses over each 1/120 s contact interval."""

    def __init__(self, poses, sizes, count, young=None, tableware=None):
        super().__init__(
            np.stack([poses, poses]),
            np.zeros(len(poses), np.int32),
            sizes,
            count,
            rate=120,
            friction=1.2,
            strength=400,
            reach=0.005,
            lifetime=1.5,
        )
        self.start = wp.zeros(1, dtype=int)
        self.previous = poses.copy()
        self.young = young
        self.tableware = tableware
        self.zero_friction = wp.zeros(count, dtype=float) if young is not None else None
        self.tangential_displacement = wp.zeros((count, len(poses)), dtype=wp.vec3)
        self.state_arrays += (self.tangential_displacement,)

    def begin_step(self, sim):
        wp.launch(
            live_colliders,
            len(self.kind),
            inputs=[
                sim.ticks,
                self.start,
                sim.dt,
                self.samples,
                self.poses,
                self.following,
            ],
        )
        wp.launch(
            sticky_contact,
            len(sim.rest),
            inputs=[
                sim.x,
                sim.v,
                sim.damage,
                self.poses,
                self.kind,
                self.sizes,
                self.attached,
                self.anchor,
                self.normals,
                self.peak,
                self.age,
                self.force,
                self.strength,
                self.reach,
                self.lifetime,
                sim.spacing,
                sim.mass,
                sim.dt,
                self.particle_weight,
            ],
        )
        wp.launch(
            adhesive_impulse,
            len(sim.rest),
            inputs=[self.attached, self.force, self.impulse, sim.dt],
        )
        if self.young is not None:
            wp.launch(
                pad_contact,
                len(sim.rest),
                inputs=[
                    sim.x,
                    sim.v,
                    self.poses,
                    self.following,
                    self.kind,
                    self.sizes,
                    self.particle_friction,
                    self.tangential_displacement,
                    self.impulse,
                    sim.mass,
                    sim.spacing,
                    self.young,
                    sim.dt,
                ],
            )

    def grid_step(self, sim):
        if self.young is None:
            super().grid_step(sim)
        if self.tableware is not None:
            self.tableware.grid_step(sim)

    def particle_step(self, sim):
        if self.young is None:
            super().particle_step(sim)
        else:
            # Last-resort nonpenetration; friction was already applied at the surface.
            wp.launch(
                particle_contact,
                len(sim.rest),
                inputs=[
                    sim.x,
                    sim.v,
                    sim.dt,
                    self.poses,
                    self.following,
                    self.kind,
                    self.sizes,
                    self.zero_friction,
                    sim.mass,
                    self.impulse,
                    self.particle_weight,
                ],
            )
        if self.tableware is not None:
            self.tableware.particle_step(sim)


class BerryRuntime:
    """Tissue solver state; Gaussian appearance never determines collision forces."""

    def __init__(self, cfg, robot, placements=None):
        self.folder = f"{cfg.asset_root}/{cfg.berry}"
        if cfg.berry_asset_version not in ("v1", "v2"):
            raise ValueError(f"Unknown berry asset version: {cfg.berry_asset_version}")
        suffix = "_v2" if cfg.berry_asset_version == "v2" else ""
        self.usd_path = cfg.berry_asset_path or f"{self.folder}/{cfg.berry}{suffix}.usdz"
        self.usd_stage, self.asset, self.proxy, self.profile = load_berry(self.usd_path)
        if self.profile["berry"] != cfg.berry:
            raise ValueError(f"Asset is for {self.profile['berry']}, but --berry selects {cfg.berry}")
        self.offset = np.array(cfg.berry_position, np.float32)
        self.fingers, _ = robot.find_bodies("panda_(left|right)finger")
        if len(self.fingers) != 2:
            raise ValueError("Expected two Franka finger bodies")
        # The two contact boxes cover the flat inner fingertip pads [m].
        self.pad_offsets = np.array([[0, 0.0054, 0.045], [0, -0.0054, 0.045]], np.float32)
        self.sizes = np.array([[0.009, 0.0054, 0.0089]] * 2, np.float32)
        wp.set_device("cuda:0")
        poses = self.collider_poses(robot)
        params = simulation_parameters(self.profile, cfg.physics_profile, cfg.mpm_hz)
        self.proxy, params, self.resolution = physics_resolution(self.proxy, params, cfg.physics_resolution)
        if placements is not None:
            self.instance_proxy, self.instance_resolution = self.proxy, self.resolution.copy()
            rotations = np.tile(np.eye(3), (len(placements), 1, 1))
            if cfg.randomize_layout and cfg.background == "ebc" and cfg.pair_layout == "plate":
                placements, rotations = random_poses(self.proxy, len(placements), cfg.layout_seed)
            self.placements, self.instance_rotations = placements, rotations
            self.proxy = combine_tissue(self.proxy, placements, rotations)
            # Separate fields must never enter the shared/cohesive velocity path.
            params.update(released=True, contact_friction=cfg.berry_friction)
            self.resolution = dict(self.resolution)
            for key in ("source_particles", "particles", "mass_kg"):
                self.resolution[key] *= len(placements)
        self.contact = LiveContact(
            poses,
            self.sizes,
            len(self.proxy["xyz"]),
            young=params["young"] if cfg.physics_profile == "handling" else None,
            tableware=TablewareContact(self.offset) if cfg.background == "ebc" else None,
        )
        grid_origin, grid_resolution = workcell_grid(cfg.background, params["h"])
        params.update(
            frame_hz=120,
            cube=False,
            contact=self.contact,
            adhesion=0,
            grid_res=grid_resolution,
        )
        if cfg.background == "ebc":
            params["plane"] = False  # The plate, bowl and actual tabletop provide support.
        self.effective_parameters = {key: value for key, value in params.items() if key != "contact"}
        self.sim = ExplicitMPM(
            self.proxy["xyz"],
            regions=self.proxy["regions"],
            interface=self.proxy["interface"],
            spacing=float(self.proxy["spacing"]),
            **params,
        )
        self.sim.origin = wp.vec3(*grid_origin)
        self.sim.prepare(0)
        self.checker = StateCheck(
            self.sim.errors,
            {
                key: getattr(self.sim, key)
                for key in (
                    "x",
                    "v",
                    "c",
                    "elastic",
                    "frames",
                    "history",
                    "damage",
                    "tear",
                    "bruise_dose",
                )
            },
            capture=True,
        )
        self.initial = {
            key: wp.clone(getattr(self.sim, key))
            for key in (
                "x",
                "v",
                "c",
                "elastic",
                "frames",
                "history",
                "damage",
                "tear",
                "bruise_dose",
                "ticks",
                "clock",
            )
        }
        self.contact_initial = [wp.clone(a) for a in self.contact.state_arrays]
        self.tick = 0
        self.last_force = np.zeros((2, 3), np.float32)

    def collider_poses(self, robot):
        """Finger collision poses in the berry simulation frame [m, xyzw]."""
        q = robot.data.body_quat_w.torch[0, self.fingers].cpu().numpy()
        p = robot.data.body_pos_w.torch[0, self.fingers].cpu().numpy()
        p = p + Rotation.from_quat(q).apply(self.pad_offsets) - self.offset
        return np.column_stack([p, q]).astype(np.float32)

    def advance(self, robot, apply_force: bool = True) -> None:
        poses = self.collider_poses(robot)
        self.contact.samples.assign(np.stack([self.contact.previous, poses]))
        self.contact.previous = poses
        self.contact.start.assign(np.array([self.tick], np.int32))
        self.contact.impulse.zero_()
        self.sim.advance(0)
        self.tick += self.sim.hz // 120
        # Equal/opposite contact impulses feed the articulated fingers next step.
        self.last_force = -120 * self.contact.impulse.numpy()
        if apply_force:
            force = torch.as_tensor(self.last_force[None], dtype=torch.float32, device=robot.device)
            robot.set_external_force_and_torque(force, torch.zeros_like(force), body_ids=self.fingers, is_global=True)

    def reset(self, robot):
        for key, value in self.initial.items():
            getattr(self.sim, key).assign(value)
        for target, value in zip(self.contact.state_arrays, self.contact_initial):
            target.assign(value)
        self.contact.previous = self.collider_poses(robot)
        self.contact.samples.assign(np.stack([self.contact.previous] * 2))
        self.contact.start.zero_()
        self.sim.errors.zero_()
        self.tick = 0
        self.last_force.fill(0)
        zero = torch.zeros((1, 2, 3), dtype=torch.float32, device=robot.device)
        robot.set_external_force_and_torque(zero, zero, body_ids=self.fingers, is_global=True)

    def metrics(self):
        self.checker.check()
        x = self.sim.x.numpy()
        metrics = {
            "center_m": (x.mean(0) + self.offset).tolist(),
            "height_m": float(np.ptp(x[:, 2])),
            "mean_damage": float(self.sim.damage.numpy().mean()),
            "max_damage": float(self.sim.damage.numpy().max()),
            "mean_tear": float(self.sim.tear.numpy().mean()),
            "tissue_span_m": np.ptp(x, axis=0).tolist(),
            "max_speed_m_s": float(np.linalg.norm(self.sim.v.numpy(), axis=1).max()),
            "finger_force_n": self.last_force.tolist(),
        }
        if self.contact.tableware is not None:
            world = x + self.offset
            inside = (np.linalg.norm(world[:, :2] - BOWL[:2], axis=1) < BOWL[3]) & (
                (world[:, 2] >= BOWL[4] - 0.001) & (world[:, 2] < BOWL[5])
            )
            metrics["fraction_in_bowl"] = float(inside.mean())
        return metrics
