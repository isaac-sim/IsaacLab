# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Convert the public BONES-SEED archive into metric G1 poses and a versioned selection manifest.

The name exclusions adapt SONIC's published filter, retaining floor acrobatics. The geometric
checks follow SuperMimic Appendix E; the unreported stationary gates are explicit below.
This independent selection is not claimed to match the authors' exact 129514-file manifest.
"""

import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import tarfile
from collections import Counter, deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab_tasks.contrib.g1_tracking.kinematics import G1_JOINT_NAMES
from isaaclab_tasks.contrib.g1_tracking.motion import CONTROL_DT, resample_motion

# Derived from NVlabs/GR00T-WholeBodyControl filter_and_copy_bones_data.py,
# blob f09061012a1f1d1da3ef35d614440629138659cd. Floor acrobatics remain included.
EXCLUDED_NAMES = (
    "bed",
    "bike",
    "chair",
    "climb",
    "com_up_50cm",
    "sitting",
    "step_on",
    "seat",
    "table",
    "sit_",
    "ladder",
    "crutch",
    "_ride_",
    "scooter",
    "stepdown",
    "box_hspu",
    "50cm_box_",
    "on_box",
    "fall_from",
    "on_1m",
    "form_box",
    "off_1m",
    "230m",
    "jump_over_obstacle_",
    "lift_crate_come_up_",
    "stair",
    "box_jump",
    "monkey_jump",
    "box_dips",
    "walking_on_edge",
    "push_obstacle",
    "jump_off_50cm",
    "look_over_fence",
)
FILTER = {
    "elevated_height_m": 0.20,
    "elevated_seconds": 0.6,
    "penetration_m": -0.12,
    "sustained_penetration_m": -0.08,
    "penetration_fraction": 0.8,
    "floating_clearance_m": 0.02,
    "floating_seconds": 0.5,
    "standing_pelvis_m": 0.55,
    "upright_cosine": 0.9,
    "stationary_max_link_speed_m_s": 0.2,
    "stationary_gate_provenance": "Independent choices; these gates are not specified numerically in the paper.",
}


def initialize_worker(mjcf_path, output, source_fps):
    """Own one MuJoCo model/data pair per conversion process; never share mutable simulator data."""
    global MODEL, DATA, FLOOR, FEET, OUTPUT, SOURCE_FPS
    torch.set_num_threads(1)
    spec = mujoco.MjSpec.from_file(mjcf_path)
    spec.worldbody.add_geom(name="filter_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
    MODEL = spec.compile()
    DATA = mujoco.MjData(MODEL)
    FLOOR = MODEL.geom("filter_floor").id
    FEET = []
    for side in ("left", "right"):
        body = MODEL.body(f"{side}_ankle_roll_link").id
        geoms = np.flatnonzero((MODEL.geom_bodyid == body) & ((MODEL.geom_contype | MODEL.geom_conaffinity) != 0))
        if len(geoms) == 0:
            raise ValueError(f"No collision geometry on {side} foot.")
        FEET.append(geoms)
    joint_names = [MODEL.joint(j).name for j in range(MODEL.njnt) if MODEL.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    if joint_names != G1_JOINT_NAMES:
        raise ValueError("Filtering model joint order differs from the pose schema.")
    OUTPUT, SOURCE_FPS = Path(output), source_fps


def convert_csv(data):
    """BONES header-defined cm/degree/Euler poses to metric xyzw/radian poses in task joint order."""
    header = next(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    expected = [
        "Frame",
        *[f"root_translate{x}" for x in "XYZ"],
        *[f"root_rotate{x}" for x in "XYZ"],
        *[f"{name}_dof" for name in G1_JOINT_NAMES],
    ]
    if len(set(header)) != len(header) or set(header) != set(expected):
        raise ValueError(f"Unexpected BONES CSV fields: {header}")
    raw = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, ndmin=2)
    raw = raw[:, [header.index(name) for name in expected]]
    if len(raw) < 4 or not np.isfinite(raw).all() or not np.allclose(np.diff(raw[:, 0]), 1, atol=0, rtol=0):
        raise ValueError("Invalid pose values or nonconsecutive source frame indices.")
    quat = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat()
    return np.concatenate((raw[:, 1:4] / 100, quat, np.deg2rad(raw[:, 7:])), axis=1).astype(np.float32)


def sustained(mask, seconds):
    n = int(round(seconds / CONTROL_DT))
    return len(mask) >= n and bool((np.convolve(mask.astype(np.int32), np.ones(n, dtype=np.int32), "valid") == n).any())


def process_motion(name, data):
    record = {"source": name, "source_sha256": hashlib.sha256(data).hexdigest()}
    raw = convert_csv(data)
    p, quat, q = resample_motion(torch.from_numpy(raw), SOURCE_FPS)
    poses = torch.cat((p, quat, q), dim=-1).numpy()
    n = len(poses)
    zmin, clearance, positions = np.empty(n), np.empty(n), []
    for i, pose in enumerate(poses):
        DATA.qpos[:] = np.concatenate((pose[:3], pose[[6, 3, 4, 5]], pose[7:]))
        mujoco.mj_kinematics(MODEL, DATA)
        zmin[i] = DATA.xpos[1:, 2].min()
        clearance[i] = min(mujoco.mj_geomDistance(MODEL, DATA, int(g), FLOOR, 10, None) for gs in FEET for g in gs)
        positions.append(DATA.xpos[1:].copy())
    max_speed = np.linalg.norm(np.gradient(np.stack(positions), CONTROL_DT, axis=0), axis=-1).max(axis=-1)
    up = Rotation.from_quat(poses[:, 3:7]).as_matrix()[:, 2, 2]
    floating = (
        (clearance > FILTER["floating_clearance_m"])
        & (poses[:, 2] > FILTER["standing_pelvis_m"])
        & (up > FILTER["upright_cosine"])
        & (max_speed < FILTER["stationary_max_link_speed_m_s"])
    )
    reason = None
    if sustained(zmin > FILTER["elevated_height_m"], FILTER["elevated_seconds"]):
        reason = "elevated_support"
    elif (zmin < FILTER["penetration_m"]).any() or (zmin < FILTER["sustained_penetration_m"]).mean() > FILTER[
        "penetration_fraction"
    ]:
        reason = "ground_penetration"
    elif sustained(floating, FILTER["floating_seconds"]):
        reason = "floating_feet"
    record.update(source_frames=len(raw), frames=n, reason=reason)
    if reason is None:
        path = Path(name).with_suffix(".npy")
        target = OUTPUT / "motions" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, poses, allow_pickle=False)
        record.update(path=path.as_posix(), pose_sha256=hashlib.sha256(poses.tobytes()).hexdigest())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--mjcf", required=True, help="Matching 29-DoF G1 MJCF from prepare_g1_tracking.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--source_fps", type=float, default=120)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty; do not mix dataset versions.")
    output.mkdir(parents=True, exist_ok=True)
    counts, kept_frames, total = Counter(), 0, 0
    manifest_path = output / "motions.jsonl"
    with (
        manifest_path.open("w") as log,
        mp.get_context("spawn").Pool(
            args.workers, initializer=initialize_worker, initargs=(args.mjcf, str(output), args.source_fps)
        ) as pool,
    ):
        pending = deque()

        def collect_one():
            nonlocal kept_frames
            result = pending.popleft().get()
            reason = result["reason"] or "kept"
            counts[reason] += 1
            if reason == "kept":
                kept_frames += result["frames"]
            log.write(json.dumps(result) + "\n")
            if sum(counts.values()) % 5000 == 0:
                log.flush()
                print("PROGRESS", dict(counts), "kept_frames", kept_frames, flush=True)

        with tarfile.open(args.archive, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError(f"Unsafe archive path: {name}")
                total += 1
                keyword = next((key for key in EXCLUDED_NAMES if key in name.stem.lower()), None)
                if keyword:
                    counts["name"] += 1
                    log.write(json.dumps({"source": member.name, "reason": "name", "keyword": keyword}) + "\n")
                    continue
                pending.append(pool.apply_async(process_motion, (member.name, archive.extractfile(member).read())))
                if len(pending) >= 2 * args.workers:
                    collect_one()
        while pending:
            collect_one()
    summary = {
        "source": "isaac/bones-seed-g1:1",
        "attribution": "Motion Data by Bones Studio",
        "raw_clips": total,
        "counts": dict(counts),
        "kept_frames": kept_frames,
        "kept_hours": (kept_frames - counts["kept"]) * CONTROL_DT / 3600,
        "source_fps": args.source_fps,
        "prepared_fps": 1 / CONTROL_DT,
        "layout": "root_xyz_m,root_quaternion_xyzw,29_joints_rad",
        "joint_names": G1_JOINT_NAMES,
        "name_exclusions": EXCLUDED_NAMES,
        "geometric_filter": FILTER,
        "exact_paper_subset": False,
        "selection_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PREPARATION_COMPLETE", json.dumps(summary), flush=True)
    if not total or not counts["kept"]:
        raise ValueError("No motions retained; inspect the selection manifest.")


if __name__ == "__main__":
    main()
