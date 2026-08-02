#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score Allegro grasp-cache quality before rotate training.

The Allegro cache row layout is:
    0:16  -> Allegro joint positions
    16:19 -> object position (x, y, z)
    19:23 -> object quaternion (x, y, z, w)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ALLEGRO_JOINT_DIM = 16
ALLEGRO_STATE_DIM = 23


def _float_triplet(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected comma-separated xyz, for example: 0.0,-0.1,0.56")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Allegro grasp-cache quality.")
    parser.add_argument("--cache", required=True, help="Path to Allegro grasp cache .npy file.")
    parser.add_argument("--expected-rows", type=int, default=50000, help="Expected minimum cache rows.")
    parser.add_argument("--state-dim", type=int, default=ALLEGRO_STATE_DIM, help="Expected row width.")
    parser.add_argument("--joint-dim", type=int, default=ALLEGRO_JOINT_DIM, help="Number of Allegro joint columns.")
    parser.add_argument(
        "--object-pos-ref",
        type=_float_triplet,
        default=None,
        help="Expected Allegro object xyz as comma-separated values. Defaults to the cache mean.",
    )
    parser.add_argument("--quat-norm-err-threshold", type=float, default=1.0e-3)
    parser.add_argument("--min-joint-std-median", type=float, default=0.03)
    parser.add_argument("--min-joint-range-median", type=float, default=0.10)
    parser.add_argument("--max-object-pos-mean-diff", type=float, default=0.08)
    parser.add_argument("--max-object-pos-p99-diff", type=float, default=0.12)
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_path = Path(args.cache).expanduser().resolve()
    if not cache_path.is_file():
        print(f"[FAIL] cache file not found: {cache_path}")
        return 2

    arr = np.load(str(cache_path))
    report: dict[str, object] = {
        "cache_path": str(cache_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "expected_rows": int(args.expected_rows),
        "state_dim": int(args.state_dim),
        "joint_dim": int(args.joint_dim),
    }

    layout_ok = args.joint_dim == ALLEGRO_JOINT_DIM and args.state_dim == args.joint_dim + 7
    shape_ok = arr.ndim == 2 and arr.shape[1] == args.state_dim and arr.shape[0] >= args.expected_rows and layout_ok
    report["layout_ok"] = bool(layout_ok)
    report["shape_ok"] = bool(shape_ok)

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite_ok = nan_count == 0 and inf_count == 0
    report["nan_count"] = nan_count
    report["inf_count"] = inf_count
    report["finite_ok"] = bool(finite_ok)

    joint_end = args.joint_dim
    obj_pos_start = joint_end
    quat_start = obj_pos_start + 3
    quat_end = quat_start + 4
    joints = arr[:, :joint_end]
    obj_pos = arr[:, obj_pos_start:quat_start]
    quat = arr[:, quat_start:quat_end]

    quat_norm = np.linalg.norm(quat, axis=1)
    quat_abs_err = np.abs(quat_norm - 1.0)
    quat_err_p99 = float(np.percentile(quat_abs_err, 99))
    quat_ok = quat_err_p99 <= args.quat_norm_err_threshold
    report["quat_order"] = "xyzw"
    report["quat_norm_mean"] = float(quat_norm.mean())
    report["quat_norm_std"] = float(quat_norm.std())
    report["quat_abs_err_p99"] = quat_err_p99
    report["quat_ok"] = bool(quat_ok)

    joint_std = joints.std(axis=0)
    joint_range = joints.max(axis=0) - joints.min(axis=0)
    joint_std_median = float(np.median(joint_std))
    joint_range_median = float(np.median(joint_range))
    diversity_ok = joint_std_median >= args.min_joint_std_median and joint_range_median >= args.min_joint_range_median
    report["joint_std_median"] = joint_std_median
    report["joint_std_min"] = float(joint_std.min())
    report["joint_range_median"] = joint_range_median
    report["joint_range_min"] = float(joint_range.min())
    report["diversity_ok"] = bool(diversity_ok)

    object_pos_ref_source = "argument"
    if args.object_pos_ref is None:
        object_pos_ref = obj_pos.mean(axis=0).astype(np.float32)
        object_pos_ref_source = "cache_mean"
    else:
        object_pos_ref = np.asarray(args.object_pos_ref, dtype=np.float32)
    object_pos_diff = np.linalg.norm(obj_pos - object_pos_ref.reshape(1, 3), axis=1)
    object_pos_mean_diff = float(object_pos_diff.mean())
    object_pos_p99_diff = float(np.percentile(object_pos_diff, 99))
    object_pos_ok = (
        object_pos_mean_diff <= args.max_object_pos_mean_diff and object_pos_p99_diff <= args.max_object_pos_p99_diff
    )
    report["object_pos_ref"] = [float(v) for v in object_pos_ref]
    report["object_pos_ref_source"] = object_pos_ref_source
    report["object_pos_mean"] = [float(v) for v in obj_pos.mean(axis=0)]
    report["object_pos_std"] = [float(v) for v in obj_pos.std(axis=0)]
    report["object_pos_min"] = [float(v) for v in obj_pos.min(axis=0)]
    report["object_pos_max"] = [float(v) for v in obj_pos.max(axis=0)]
    report["object_pos_mean_diff"] = object_pos_mean_diff
    report["object_pos_p99_diff"] = object_pos_p99_diff
    report["object_pos_ok"] = bool(object_pos_ok)

    row_score = min(1.0, arr.shape[0] / max(float(args.expected_rows), 1.0))
    diversity_score = 0.5 * min(1.0, joint_std_median / max(args.min_joint_std_median, 1.0e-6))
    diversity_score += 0.5 * min(1.0, joint_range_median / max(args.min_joint_range_median, 1.0e-6))
    quat_score = 1.0 if quat_err_p99 <= 1.0e-6 else max(0.0, 1.0 - quat_err_p99 / args.quat_norm_err_threshold)
    pos_score = 0.5 * max(0.0, 1.0 - object_pos_mean_diff / max(args.max_object_pos_mean_diff, 1.0e-6))
    pos_score += 0.5 * max(0.0, 1.0 - object_pos_p99_diff / max(args.max_object_pos_p99_diff, 1.0e-6))
    score = 100.0 * (0.30 * row_score + 0.25 * diversity_score + 0.20 * quat_score + 0.25 * pos_score)
    report["score"] = float(score)

    train_readiness = shape_ok and finite_ok and quat_ok and diversity_ok and object_pos_ok
    report["train_readiness"] = "PASS" if train_readiness else "FAIL"

    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(f"[INFO] cache_path: {report['cache_path']}")
        print(f"[INFO] shape={tuple(report['shape'])} dtype={report['dtype']}")
        print(
            "[INFO] checks: "
            f"layout_ok={layout_ok} shape_ok={shape_ok} finite_ok={finite_ok} "
            f"quat_ok={quat_ok} diversity_ok={diversity_ok} object_pos_ok={object_pos_ok}"
        )
        print(
            "[INFO] metrics: "
            f"joint_std_median={joint_std_median:.6f}, "
            f"joint_range_median={joint_range_median:.6f}, "
            f"quat_abs_err_p99={quat_err_p99:.6e}, "
            f"object_pos_mean_diff={object_pos_mean_diff:.6f}, "
            f"object_pos_p99_diff={object_pos_p99_diff:.6f}"
        )
        print(f"[RESULT] score={score:.2f} train_readiness={report['train_readiness']}")

    return 0 if train_readiness else 1


if __name__ == "__main__":
    raise SystemExit(main())
