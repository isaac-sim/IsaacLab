# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identical tissues in a shared MPM grid, with separate rendering views."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from ..assets.sh_rotation import rotate_sh
from ..scene.tableware import BOWL, PUNNET

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..pick_berries_env_cfg import BerryPickEnvCfg
    from .runtime import BerryRuntime


def combine_tissue(proxy: dict, offsets: np.ndarray, rotations: np.ndarray | None = None) -> dict:
    """Duplicate tissue into two or three independent material fields at offsets [m]."""
    offsets = np.asarray(offsets, np.float32)
    if offsets.shape not in ((2, 3), (3, 3)) or not np.isfinite(offsets).all():
        raise ValueError("A berry group requires two or three finite XYZ offsets")
    if len(np.unique(proxy["regions"])) != 1:
        raise ValueError("A coupled berry group requires a single-region source tissue")
    count = len(proxy["xyz"])
    if rotations is None:
        rotations = np.tile(np.eye(3), (len(offsets), 1, 1))
    rotations = np.asarray(rotations)
    if rotations.shape != (len(offsets), 3, 3) or not np.isfinite(rotations).all():
        raise ValueError("Each berry requires one finite 3x3 rotation matrix")
    return {
        "xyz": np.concatenate([proxy["xyz"] @ r.T + shift for r, shift in zip(rotations, offsets)]).astype(np.float32),
        "regions": np.repeat(np.arange(len(offsets), dtype=np.int32), count),
        "interface": np.tile(proxy["interface"], len(offsets)),
        "spacing": proxy["spacing"],
    }


def random_poses(proxy: dict, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample separated centers [m] and full 3D orientations inside the punnet.

    The returned affine translations put each rotated tissue's bottom on its
    support plane. Extra rim clearance permits a vertical gripper approach.
    """
    if count not in (2, 3):
        raise ValueError("Random placement requires two or three berries")
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(count, random_state=rng).as_matrix()
    center = (proxy["xyz"].min(0) + proxy["xyz"].max(0)) / 2
    rotated = [(proxy["xyz"] - center) @ r.T for r in rotations]
    radius = max(float(np.linalg.norm(p[:, :2], axis=1).max()) for p in rotated)
    bounds = np.array(PUNNET[2:4]) - radius - 0.016
    if np.any(bounds <= 0):
        raise ValueError("Two or three berries must fit inside the punnet with gripper clearance")
    for _ in range(10000):
        centers = rng.uniform(-bounds, bounds, size=(count, 2))
        if all(np.linalg.norm(centers[i] - centers[j]) >= 2 * radius + 0.014 for i in range(count) for j in range(i)):
            break
    else:
        raise ValueError("Unable to fit separated berries with gripper clearance in the punnet")
    centers = np.asarray(sorted(centers, key=lambda p: p[1]))
    shifts = []
    for r, p, xy in zip(rotations, rotated, centers):
        shifts.append(np.array([*xy, -p[:, 2].min()]) - r @ center)
    return np.asarray(shifts, np.float32), rotations


def pair_offsets(cfg: BerryPickEnvCfg) -> np.ndarray:
    """Initial per-instance translations relative to the common task origin [m]."""
    if cfg.berry != "raspberry":
        raise ValueError("Coupled groups currently support raspberries only")
    if cfg.berry_count == 3:
        if cfg.pair_layout != "plate":
            raise ValueError("Three raspberries require the plate layout")
        return np.array([[0, -0.018, 0], [-0.024, 0.018, 0], [0.024, 0.018, 0]], np.float32)
    if cfg.pair_layout == "plate":
        return np.array([[-0.022, 0, 0], [0.022, 0, 0]], np.float32)
    if cfg.pair_layout == "drop":
        return np.array([[0, 0, 0.04], [0, 0, 0]], np.float32)
    if cfg.pair_layout == "bowl" and cfg.background == "ebc":
        return np.array([[0, 0, 0], np.array([*BOWL[:2], BOWL[4]]) - cfg.berry_position], np.float32)
    raise ValueError("Use pair layout plate/drop, or bowl with --background ebc")


class BerryInstance:
    """Zero-copy particle view of one berry; its owner advances and resets the group."""

    def __init__(self, owner: BerryRuntime, index: int, shift: np.ndarray):
        self.owner = owner
        self.mpm_device = owner.mpm_device
        count = len(owner.instance_proxy["xyz"])
        selection = slice(index * count, (index + 1) * count)
        self.usd_path, self.usd_stage, self.profile = owner.usd_path, owner.usd_stage, owner.profile
        rotation = owner.instance_rotations[index]
        self.asset = dict(owner.asset, xyz=(owner.asset["xyz"] @ rotation.T + shift).astype(np.float32))
        self.asset["rotations"] = (
            (Rotation.from_matrix(rotation) * Rotation.from_quat(owner.asset["rotations"])).as_quat().astype(np.float32)
        )
        self.asset["sh"] = rotate_sh(owner.asset["sh"], rotation)
        self.proxy = dict(
            owner.instance_proxy, xyz=(owner.instance_proxy["xyz"] @ rotation.T + shift).astype(np.float32)
        )
        self.offset = owner.offset
        self.initial_position = owner.offset + self.proxy["xyz"].mean(0)
        self.initial_rotation_xyzw = Rotation.from_matrix(rotation).as_quat()
        self.resolution = owner.instance_resolution
        self.effective_parameters = owner.effective_parameters
        self.sizes = owner.sizes
        self.contact, self.checker = owner.contact, owner.checker
        self.sim = SimpleNamespace(
            rest=owner.sim.rest[selection],
            hz=owner.sim.hz,
            cfl=owner.sim.cfl,
            **{key: getattr(owner.sim, key)[selection] for key in ("x", "v", "damage", "tear")},
        )

    @property
    def last_force(self) -> np.ndarray:
        """Total finger reaction of the coupled group [N], not an instance contribution."""
        return self.owner.last_force

    def collider_poses(self, robot: Articulation) -> np.ndarray:
        """Finger collision poses in the shared simulation frame [m, xyzw]."""
        return self.owner.collider_poses(robot)

    def metrics(self) -> dict:
        """Per-instance state with explicitly labelled total group finger reaction."""
        from .runtime import BerryRuntime

        result = BerryRuntime.metrics(self)
        result["finger_force_scope"] = "coupled_pair_total" if self.owner.sim.fields == 2 else "coupled_group_total"
        return result
