# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless pad/tissue benchmark; isolates contact from robot IK and rendering."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from isaaclab_tasks.contrib.franka_pick_berries.assets.asset_root import berry_root
from isaaclab_tasks.contrib.franka_pick_berries.assets.usd_asset import load_berry
from isaaclab_tasks.contrib.franka_pick_berries.physics.materials import simulation_parameters
from isaaclab_tasks.contrib.franka_pick_berries.physics.mpm.explicit_mpm import ExplicitMPM
from isaaclab_tasks.contrib.franka_pick_berries.physics.resolution import physics_resolution
from isaaclab_tasks.contrib.franka_pick_berries.physics.runtime import LiveContact


def main():
    """Close, lift/hold/release or compress using physical pad contact [SI units]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--berry", choices=["raspberry", "blackberry", "blueberry", "strawberry"], default="raspberry")
    parser.add_argument("--asset_version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--physics_profile", choices=["handling", "legacy"], default="handling")
    parser.add_argument("--physics_resolution", choices=["full", "half"], default="full")
    parser.add_argument("--mode", choices=["pick", "squash", "compression"], default="pick")
    parser.add_argument("--gap", type=float, help="Final inner-pad gap [m]; default: 85%% of settled width")
    parser.add_argument("--grid_shift", type=float, default=0, help="Shift only the grid origin along Y [m]")
    parser.add_argument("--friction", type=float, default=1.2, help="Finger Coulomb friction coefficient")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; existing runs are never overwritten")
    if not np.isfinite([args.grid_shift, args.friction]).all() or args.friction < 0:
        parser.error("Require finite grid shift and nonnegative finite friction")
    if args.gap is not None and not 0 < args.gap <= 0.08:
        parser.error("--gap must be in (0, 0.08] m")
    suffix = "_v2" if args.asset_version == "v2" else ""
    asset_path = f"{berry_root()}/{args.berry}/{args.berry}{suffix}.usdz"
    _, _, proxy, profile = load_berry(asset_path)
    params = simulation_parameters(profile, args.physics_profile)
    proxy, params, resolution = physics_resolution(proxy, params, args.physics_resolution)
    wp.config.enable_backward = False
    wp.set_device("cuda:0")
    rest = proxy["xyz"]
    center = (rest.min(0) + rest.max(0)) / 2
    sizes = np.array([[0.009, 0.0054, 0.0089]] * 2, np.float32)
    poses = np.array(
        [[center[0], -0.045, center[2], 0, 0, 0, 1], [center[0], 0.045, center[2], 0, 0, 0, 1]], np.float32
    )
    if args.mode == "compression":
        sizes = np.array([[0.009, 0.009, 0.009]], np.float32)
        poses = np.array([[center[0], center[1], rest[:, 2].max() + 0.010, 0, 0, 0, 1]], np.float32)
    contact = LiveContact(
        poses, sizes, len(rest), young=params["young"] if args.physics_profile == "handling" else None
    )
    contact.field_friction.fill_(args.friction)
    contact.particle_friction.fill_(args.friction)
    params.update(cube=False, adhesion=0, frame_hz=120, grid_res=(121, 121, 121))
    sim = ExplicitMPM(
        rest,
        regions=proxy["regions"],
        interface=proxy["interface"],
        spacing=float(proxy["spacing"]),
        contact=contact,
        **params,
    )
    sim.origin = wp.vec3(-0.12, -0.12 + args.grid_shift, -0.008)
    sim.prepare(0)
    args.output.mkdir(parents=True)
    rows = []
    gap = 0.001 if args.mode == "squash" else args.gap
    started = time.perf_counter()
    num_steps = 120 * 12
    for step in range(num_steps):
        t = (step + 1) / 120
        if step == 120:
            settled = sim.x.numpy()
            center = (settled.min(0) + settled.max(0)) / 2
            if gap is None:
                gap = float(np.ptp(settled[:, 1]) * 0.85)
        following = poses.copy()
        if args.mode == "compression":
            opening = rest[:, 2].max() * (1 - 0.8 * np.clip((t - 1) / 8, 0, 1))
            following[0, :2] = center[:2]
            following[0, 2] = opening + sizes[0, 2]
        else:
            opening = 0.08 + ((gap or 0.02) - 0.08) * np.clip((t - 1) / 3, 0, 1)
            if t > 9:
                opening += (0.08 - gap) * np.clip(t - 9, 0, 1)
            following[:, 0] = center[0]
            following[:, 1] = center[1] + np.array([-1, 1]) * (opening / 2 + sizes[:, 1])
            following[:, 2] = center[2] + (0.05 * np.clip((t - 5) / 2, 0, 1) if args.mode == "pick" else 0)
        contact.samples.assign(np.stack([poses, following]))
        contact.start.assign(np.array([step * (sim.hz // 120)], np.int32))
        contact.impulse.zero_()
        sim.advance(0)
        poses = following
        if step % 30 == 29:
            sim.check()
            x = sim.x.numpy()
            rows.append(
                dict(
                    time_s=t,
                    opening_m=float(opening),
                    center_m=x.mean(0).tolist(),
                    tissue_span_m=np.ptp(x, axis=0).tolist(),
                    minimum_z_m=float(x[:, 2].min()),
                    mean_damage=float(sim.damage.numpy().mean()),
                    mean_tear=float(sim.tear.numpy().mean()),
                    finger_force_n=(-120 * contact.impulse.numpy()).tolist(),
                )
            )
    elapsed = time.perf_counter() - started
    baseline = rows[3]["center_m"][2]
    hold = [r for r in rows if 8 <= r["time_s"] <= 9]
    summary = dict(
        minimum_hold_lift_m=min(r["center_m"][2] for r in hold) - baseline,
        hold_drift_m=abs(hold[-1]["center_m"][2] - hold[0]["center_m"][2]),
        final_lift_m=rows[-1]["center_m"][2] - baseline,
        hold_mean_damage=max(r["mean_damage"] for r in hold),
        maximum_mean_damage=max(r["mean_damage"] for r in rows),
        maximum_mean_tear=max(r["mean_tear"] for r in rows),
        wall_seconds=elapsed,
        physics_frames_per_second=num_steps / elapsed,
    )
    report = dict(
        berry=args.berry,
        asset=str(asset_path),
        physics_profile=args.physics_profile,
        physics_resolution=resolution,
        mode=args.mode,
        grid_shift_m=args.grid_shift,
        friction=args.friction,
        gap_m=gap,
        parameters=params,
        cfl=sim.cfl,
        mass_kg=sim.mass * len(rest),
        summary=summary,
        rows=rows,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
