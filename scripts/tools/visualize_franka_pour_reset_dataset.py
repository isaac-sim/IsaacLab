# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render every Franka Pour reset state into four overlaid PNGs.

This is an offline dataset diagnostic: it uses Pinocchio and Isaac Lab's Franka
URDF to reconstruct the robot link poses, so it does not launch Kit, Newton, or
a GPU simulation. Every state is rendered; the script never subsamples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pinocchio as pin
import torch

from isaaclab_tasks.contrib.franka_pour.reset_dataset_generator import validate_reset_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "datasets/franka_pour/reset_dataset.pt"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "datasets/franka_pour/reset_dataset_visualization"
_DEFAULT_URDF = _REPO_ROOT / "source/isaaclab/isaaclab/controllers/config/data/lula_franka_gen.urdf"

_ROBOT_FRAMES = tuple(f"panda_link{index}" for index in range(8)) + (
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
)
_ROBOT_EDGES = np.asarray(tuple((index, index + 1) for index in range(8)) + ((8, 9), (8, 10)))
_BOX_EDGES = np.asarray(
    (
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
)


def _objective_buckets(objective: torch.Tensor) -> tuple[tuple[str, str, torch.Tensor, str], ...]:
    """Return the requested four disjoint objective buckets."""
    buckets = (
        ("minus_1", "Non-grasping states: J = -1", objective == -1.0, "#58677c"),
        ("0_to_0p5", "Grasping states: 0 ≤ J < 0.5", (objective >= 0.0) & (objective < 0.5), "#276fbf"),
        (
            "0p5_to_0p9",
            "Grasping states: 0.5 ≤ J < 0.9",
            (objective >= 0.5) & (objective < 0.9),
            "#d98e04",
        ),
        (
            "0p9_to_1",
            "Grasping states: 0.9 ≤ J ≤ 1",
            (objective >= 0.9) & (objective <= 1.0),
            "#c43832",
        ),
    )
    assignment_count = torch.stack([mask for _, _, mask, _ in buckets]).sum(dim=0)
    if not bool((assignment_count == 1).all()):
        invalid = objective[assignment_count != 1]
        raise ValueError(
            f"Every state must belong to exactly one objective bucket; found {invalid.numel()} invalid values."
        )
    return buckets


def _load_dataset(path: Path) -> dict:
    """Load the dataset and validate the fields needed by this diagnostic."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_reset_dataset(payload)
    try:
        states = payload["states"]
        contract = payload["metadata"]["task_contract"]
        required_states = {
            "arm_joint_position": (7,),
            "finger_joint_position": (2,),
            "source_root_pose": (7,),
            "target_root_pose": (7,),
            "objective": (),
        }
        state_count = int(states["objective"].shape[0])
        for name, trailing_shape in required_states.items():
            value = states[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (state_count, *trailing_shape):
                raise ValueError(f"Invalid {name!r} tensor shape: expected {(state_count, *trailing_shape)}.")
        for name in ("source_box_half", "target_box_half"):
            if len(contract[name]) != 3:
                raise ValueError(f"Invalid task-contract field {name!r}.")
    except KeyError as error:
        raise ValueError(f"Dataset is missing required field {error.args[0]!r}.") from error

    if state_count == 0 or not bool(torch.isfinite(states["objective"]).all()):
        raise ValueError("The dataset must contain finite objective values and at least one state.")
    _objective_buckets(states["objective"])
    return payload


def _robot_frame_positions(
    urdf_path: Path,
    arm_joint_position: torch.Tensor,
    finger_joint_position: torch.Tensor,
) -> np.ndarray:
    """Evaluate all dataset Franka frame positions with Pinocchio FK."""
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    q = pin.neutral(model)
    joint_names = tuple(f"panda_joint{index}" for index in range(1, 8)) + (
        "panda_finger_joint1",
        "panda_finger_joint2",
    )
    joint_indices = np.asarray([model.joints[model.getJointId(name)].idx_q for name in joint_names])
    frame_ids = tuple(model.getFrameId(name) for name in _ROBOT_FRAMES)
    if any(frame_id >= len(model.frames) for frame_id in frame_ids):
        raise RuntimeError("The Franka URDF does not contain every expected robot frame.")

    joint_position = torch.cat((arm_joint_position, finger_joint_position), dim=1).numpy()
    positions = np.empty((joint_position.shape[0], len(frame_ids), 3), dtype=np.float32)
    for row, cached_q in enumerate(joint_position):
        q[joint_indices] = cached_q
        pin.framesForwardKinematics(model, data, q)
        positions[row] = np.asarray([data.oMf[frame_id].translation for frame_id in frame_ids])
    return positions


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a batch of normalized XYZW quaternions to rotation matrices."""
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=1.0e-12)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)


def _box_segments(poses: torch.Tensor, half_extents: tuple[float, float, float]) -> np.ndarray:
    """Return world-space edges of bottom-origin oriented collision-proxy boxes."""
    half_x, half_y, half_z = half_extents
    corners = np.asarray(
        tuple((x, y, z) for z in (0.0, 2.0 * half_z) for y in (-half_y, half_y) for x in (-half_x, half_x)),
        dtype=np.float32,
    )
    poses_np = poses.numpy()
    rotation = _quat_xyzw_to_matrix(poses_np[:, 3:7])
    world_corners = np.einsum("nij,kj->nki", rotation, corners) + poses_np[:, None, :3]
    return world_corners[:, _BOX_EDGES]


def _table_grid(lower_xy: np.ndarray, upper_xy: np.ndarray, cells: int = 10) -> np.ndarray:
    """Create a z=0 grid spanning the actual tabletop support footprint."""
    x_values = np.linspace(lower_xy[0], upper_xy[0], cells + 1)
    y_values = np.linspace(lower_xy[1], upper_xy[1], cells + 1)
    segments = [((x, lower_xy[1], 0.0), (x, upper_xy[1], 0.0)) for x in x_values]
    segments += [((lower_xy[0], y, 0.0), (upper_xy[0], y, 0.0)) for y in y_values]
    return np.asarray(segments, dtype=np.float32)


def _common_bounds(*point_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed plot bounds shared by all four output images."""
    points = np.concatenate([value.reshape(-1, 3) for value in point_sets], axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    lower = np.minimum(lower, np.asarray((-0.15, -0.70, 0.0), dtype=np.float32))
    upper = np.maximum(upper, np.asarray((0.90, 0.70, 0.85), dtype=np.float32))
    padding = np.maximum(0.035 * (upper - lower), np.asarray((0.02, 0.02, 0.015)))
    return lower - padding, upper + padding


def _render_bucket(
    output_path: Path,
    title: str,
    mask: np.ndarray,
    robot_color: str,
    robot_segments: np.ndarray,
    source_segments: np.ndarray,
    target_segments: np.ndarray,
    table_segments: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    dpi: int,
    azimuth: float,
    elevation: float,
    content_sha256: str,
) -> None:
    """Render one objective bucket with every matching state overlaid."""
    count = int(mask.sum())
    alpha = min(0.72, max(0.006, 12.0 / max(count, 1)))
    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    axis.add_collection3d(Line3DCollection(table_segments, colors="#9da3aa", linewidths=0.7, alpha=0.65))
    for segments, color, width, alpha_scale, zorder in (
        (robot_segments, robot_color, 0.55, 1.0, 3),
        (source_segments, "#e56b2f", 0.40, 0.85, 4),
        (target_segments, "#159a9c", 0.40, 0.85, 2),
    ):
        selected = segments[mask].reshape(-1, 2, 3)
        axis.add_collection3d(
            Line3DCollection(
                selected,
                colors=color,
                linewidths=width,
                alpha=min(1.0, alpha * alpha_scale),
                rasterized=True,
                zorder=zorder,
            )
        )

    lower, upper = bounds
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect(upper - lower)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_proj_type("ortho")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.grid(False)
    axis.xaxis.pane.set_alpha(0.0)
    axis.yaxis.pane.set_alpha(0.0)
    axis.zaxis.pane.set_alpha(0.0)
    axis.set_title(f"{title}\n{count:,} states — all overlaid, no subsampling", fontsize=16, fontweight="semibold")
    axis.legend(
        handles=(
            Line2D((0,), (0,), color=robot_color, linewidth=2.0, label="Franka link skeleton"),
            Line2D((0,), (0,), color="#e56b2f", linewidth=2.0, label="Source cup collision proxy"),
            Line2D((0,), (0,), color="#159a9c", linewidth=2.0, label="Target bowl collision proxy"),
            Line2D((0,), (0,), color="#9da3aa", linewidth=2.0, label="Table plane"),
        ),
        loc="upper left",
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=dpi,
        facecolor="white",
        metadata={
            "Title": title,
            "Description": f"All {count} matching states; dataset content SHA-256: {content_sha256}",
        },
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_DEFAULT_DATASET, help="Validated reset dataset.")
    parser.add_argument("--output_dir", type=Path, default=_DEFAULT_OUTPUT_DIR, help="Directory for the four PNGs.")
    parser.add_argument("--urdf", type=Path, default=_DEFAULT_URDF, help="Franka URDF used for forward kinematics.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--camera_azimuth", type=float, default=-55.0)
    parser.add_argument("--camera_elevation", type=float, default=26.0)
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    dataset_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    payload = _load_dataset(dataset_path)
    states = payload["states"]
    contract = payload["metadata"]["task_contract"]
    robot_positions = _robot_frame_positions(
        urdf_path,
        states["arm_joint_position"],
        states["finger_joint_position"],
    )
    robot_segments = robot_positions[:, _ROBOT_EDGES]
    source_segments = _box_segments(states["source_root_pose"], tuple(contract["source_box_half"]))
    target_segments = _box_segments(states["target_root_pose"], tuple(contract["target_box_half"]))
    bounds = _common_bounds(robot_segments, source_segments, target_segments)
    table_segments = _table_grid(
        np.asarray(contract["tabletop_support_lower_xy"], dtype=np.float32),
        np.asarray(contract["tabletop_support_upper_xy"], dtype=np.float32),
    )

    outputs = []
    buckets = _objective_buckets(states["objective"])
    for slug, title, mask, robot_color in buckets:
        output_path = output_dir / f"reset_states_{slug}.png"
        _render_bucket(
            output_path,
            title,
            mask.numpy(),
            robot_color,
            robot_segments,
            source_segments,
            target_segments,
            table_segments,
            bounds,
            dpi=args.dpi,
            azimuth=args.camera_azimuth,
            elevation=args.camera_elevation,
            content_sha256=payload["content_sha256"],
        )
        outputs.append(output_path)

    counts = [int(mask.sum()) for _, _, mask, _ in buckets]
    print(f"[INFO] Rendered all {sum(counts):,} reset states without subsampling.")
    print(f"[INFO] Bucket counts (-1, [0,.5), [.5,.9), [.9,1]): {counts}.")
    for output_path in outputs:
        print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
