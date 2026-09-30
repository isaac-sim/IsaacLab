# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless two-raspberry contact probes, including a contact-disabled control."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from isaaclab_tasks.contrib.franka_pick_berries.assets.asset_root import berry_root
from isaaclab_tasks.contrib.franka_pick_berries.assets.usd_asset import load_berry
from isaaclab_tasks.contrib.franka_pick_berries.physics.materials import simulation_parameters
from isaaclab_tasks.contrib.franka_pick_berries.physics.mpm.explicit_mpm import ExplicitMPM
from isaaclab_tasks.contrib.franka_pick_berries.physics.pair import combine_tissue
from isaaclab_tasks.contrib.franka_pick_berries.physics.resolution import physics_resolution


def main() -> None:
    """Measure free pushing, separation or a gravity-driven drop in SI units."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["push", "separate", "drop"], default="push")
    parser.add_argument("--contact", choices=["on", "off"], default="on")
    parser.add_argument("--physics_resolution", choices=["full", "half"], default="half")
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    if not np.isfinite([args.seconds, args.friction]).all() or args.seconds <= 0 or args.friction < 0:
        parser.error("Require positive finite duration and nonnegative finite friction")
    _, _, proxy, profile = load_berry(f"{berry_root()}/raspberry/raspberry_v2.usdz")
    params = simulation_parameters(profile, "handling")
    proxy, params, resolution = physics_resolution(proxy, params, args.physics_resolution)
    count = len(proxy["xyz"])
    shifts = np.array([[-0.025, 0, 0], [0.025, 0, 0]], np.float32)
    velocities = np.zeros((2 * count, 3), np.float32)
    if args.case == "push":
        velocities[:count, 0] = 0.08
    elif args.case == "separate":
        shifts[:, 0] = [-0.012, 0.012]
        velocities[:count, 0], velocities[count:, 0] = -0.02, 0.02
    else:
        shifts = np.array([[0, 0, 0.04], [0, 0, 0]], np.float32)
    merged = combine_tissue(proxy, shifts)
    params.update(
        released=True,
        contact_friction=args.friction,
        interfield_contact=args.contact == "on",
        gravity=(0, 0, -9.81) if args.case == "drop" else (0, 0, 0),
        damping=1.0 if args.case == "drop" else 0.0,
        plane=args.case == "drop",
        cube=False,
        frame_hz=120,
        grid_res=(121, 121, 121),
    )
    wp.config.enable_backward = False
    sim = ExplicitMPM(
        merged["xyz"],
        regions=merged["regions"],
        interface=merged["interface"],
        spacing=float(proxy["spacing"]),
        **params,
    )
    sim.origin = wp.vec3(-0.08, -0.08, -0.008)
    sim.v.assign(velocities)
    sim.prepare(0)
    rows = []

    def sample(t):
        sim.check()
        x, v = sim.x.numpy(), sim.v.numpy()
        rows.append(
            dict(
                time_s=t,
                centers_m=[x[:count].mean(0).tolist(), x[count:].mean(0).tolist()],
                velocities_m_s=[v[:count].mean(0).tolist(), v[count:].mean(0).tolist()],
                momentum_kg_m_s=(sim.mass * v.sum(0)).tolist(),
                kinetic_energy_j=float(0.5 * sim.mass * np.square(v).sum()),
                closest_particles_m=float(cKDTree(x[count:]).query(x[:count])[0].min()),
                height_m=[float(np.ptp(x[:count, 2])), float(np.ptp(x[count:, 2]))],
                mean_damage=[float(sim.damage.numpy()[:count].mean()), float(sim.damage.numpy()[count:].mean())],
            )
        )

    sample(0)
    started = time.perf_counter()
    for step in range(round(args.seconds * 120)):
        sim.advance(0)
        if step % 12 == 11:
            sample((step + 1) / 120)
    elapsed = time.perf_counter() - started
    report = dict(
        case=args.case,
        contact=args.contact,
        friction=args.friction,
        resolution=resolution,
        mass_per_berry_kg=sim.mass * count,
        wall_seconds=elapsed,
        rows=rows,
    )
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows[-1], indent=2))


if __name__ == "__main__":
    main()
