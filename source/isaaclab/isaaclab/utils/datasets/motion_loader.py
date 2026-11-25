# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2026 ETH Zurich, Chong Zhang.

"""Load named reference arrays once, in the runtime robot's joint and body order."""

import csv
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
)


@configclass
class CsvMotionCfg:
    """Column and coordinate conventions for read-only robot pose CSV input.

    Columns are zero-based indices or names in the header. Joint mappings use
    the target robot's joint names as keys. Unmapped joints use consecutive
    columns after the root pose for headerless files, or ``joint_column_template``
    for named files. Each file is one clip; no rows are filtered or written back.
    """

    header: bool = False
    delimiter: str = ","
    root_position: list[str | int] = [0, 1, 2]
    root_rotation: list[str | int] = [3, 4, 5, 6]
    rotation_format: str = "xyzw"
    """Root rotation: ``xyzw``, ``wxyz``, or extrinsic ``euler_xyz``."""
    joint_columns: dict[str, str | int] = {}
    joint_column_template: str = "{name}"
    position_scale: float = 1.0
    """Source length unit in meters, e.g. 0.01 for centimeters."""
    angles_in_degrees: bool = False
    """Applies to joint angles, Euler angles and supplied angular velocities."""
    world_rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """XYZW rotation from the source world frame to the training world frame.

    This rotates global poses and velocities; it does not retarget the robot's
    local link frames or change its joint axes.
    """
    root_linear_velocity: list[str | int] | None = None
    root_angular_velocity: list[str | int] | None = None
    """Optional root angular velocity columns, expressed in the source world frame."""
    joint_velocity_columns: dict[str, str | int] | None = None
    """Optional velocity columns for every target joint, per second."""


def load_csv_motion(path: str | Path, joint_names: Sequence[str], cfg: CsvMotionCfg) -> dict[str, torch.Tensor]:
    """Read one clip into CPU float32 pose/velocity tensors without changing the source.

    Root quaternions use Isaac Lab's XYZW convention. Root and joint poses are
    required; supplied velocities are optional and are not recomputed here.
    Headerless defaults read XYZ/XYZW/joint-radian poses, without a frame column.
    """
    if not np.isfinite(cfg.position_scale) or cfg.position_scale <= 0:
        raise ValueError("CSV position_scale must be finite and positive.")
    if cfg.rotation_format not in ("xyzw", "wxyz", "euler_xyz"):
        raise ValueError(f"Unsupported CSV rotation format: {cfg.rotation_format}.")
    if len(set(joint_names)) != len(joint_names) or set(cfg.joint_columns) - set(joint_names):
        raise ValueError("CSV joint names must be unique and mappings must name target joints.")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter=cfg.delimiter)) if cfg.header else None
        if header is not None and len(header) != len(set(header)):
            raise ValueError(f"Duplicate CSV column names in {path}.")
        raw = np.loadtxt(stream, delimiter=cfg.delimiter, ndmin=2, comments=None)
    if len(raw) < 2:
        raise ValueError(f"Motion {path} must have at least two frames.")

    def select(columns, width):
        if len(columns) != width:
            raise ValueError(f"Expected {width} CSV columns, got {columns}.")
        indices = []
        for column in columns:
            if isinstance(column, str):
                if header is None or column not in header:
                    raise ValueError(f"Missing CSV header column {column!r} in {path}.")
                column = header.index(column)
            if not isinstance(column, int) or not 0 <= column < raw.shape[1]:
                raise ValueError(f"Invalid CSV column {column!r} in {path}.")
            indices.append(column)
        if len(indices) != len(set(indices)):
            raise ValueError(f"Repeated CSV columns: {columns}.")
        values = torch.tensor(raw[:, indices], dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError(f"Nonfinite motion values in {path}.")
        return values

    rotation_width = 3 if cfg.rotation_format == "euler_xyz" else 4
    joint_columns = [
        cfg.joint_columns.get(
            name, cfg.joint_column_template.format(name=name) if cfg.header else 3 + rotation_width + i
        )
        for i, name in enumerate(joint_names)
    ]
    angle_scale = np.pi / 180 if cfg.angles_in_degrees else 1.0
    rotation = select(cfg.root_rotation, rotation_width)
    if cfg.rotation_format == "euler_xyz":
        rotation = quat_from_euler_xyz(*(rotation * angle_scale).unbind(-1))
    elif cfg.rotation_format == "wxyz":
        rotation = rotation[:, [1, 2, 3, 0]]
    if (rotation.norm(dim=-1) < 1e-6).any():
        raise ValueError(f"Invalid root quaternion in {path}.")
    motion = {
        "root_pos_w": select(cfg.root_position, 3) * cfg.position_scale,
        "root_quat_w": rotation,
        "joint_pos": select(joint_columns, len(joint_names)) * angle_scale,
    }
    for field, columns, scale in (
        ("root_lin_vel_w", cfg.root_linear_velocity, cfg.position_scale),
        ("root_ang_vel_w", cfg.root_angular_velocity, angle_scale),
    ):
        if columns is not None:
            motion[field] = select(columns, 3) * scale
    if cfg.joint_velocity_columns is not None:
        if set(cfg.joint_velocity_columns) != set(joint_names):
            raise ValueError("CSV joint velocity columns must cover every target joint.")
        motion["joint_vel"] = (
            select([cfg.joint_velocity_columns[name] for name in joint_names], len(joint_names)) * angle_scale
        )
    world = torch.tensor(cfg.world_rotation, dtype=torch.float32)
    if (
        world.shape != (4,)
        or not torch.isfinite(world).all()
        or not torch.isclose(world.norm(), world.new_tensor(1.0), atol=1e-6)
    ):
        raise ValueError("CSV world_rotation must be a unit XYZW quaternion.")
    if tuple(cfg.world_rotation) != (0.0, 0.0, 0.0, 1.0):
        world = world.expand(len(raw), -1)
        motion["root_quat_w"] = quat_mul(world, motion["root_quat_w"])
        for field in ("root_pos_w", "root_lin_vel_w", "root_ang_vel_w"):
            if field in motion:
                motion[field] = quat_apply(world, motion[field])
    return motion


def resample_motion(motion: dict[str, torch.Tensor], source_fps: float, target_fps: float) -> dict[str, torch.Tensor]:
    """Resample one uniformly timed clip in memory; quaternions are XYZW.

    Interpolate poses and any supplied velocities without crossing clip boundaries.
    At equal rates, preserve supplied values except for quaternion normalization.
    """
    if not all(np.isfinite(rate) and rate > 0 for rate in (source_fps, target_fps)):
        raise ValueError("Motion frame rates must be finite and positive.")
    quat = motion["root_quat_w"]
    if len(quat) < 2 or any(len(value) != len(quat) for value in motion.values()):
        raise ValueError("Motion arrays must share at least two frames.")
    if source_fps == target_fps:
        return {**motion, "root_quat_w": F.normalize(quat, dim=-1)}
    dt = 1.0 / target_fps
    n = int(np.floor(((len(quat) - 1) / source_fps) / dt)) + 1
    if n < 2:
        raise ValueError("Motion is shorter than one control transition.")
    t = torch.arange(n, dtype=torch.float32, device=quat.device) * dt * source_fps
    i0 = torch.clamp(t.floor().long(), max=len(quat) - 2)
    weight = (t - i0.float()).clamp(0, 1)[:, None]
    result = {
        key: value[i0] * (1 - weight) + value[i0 + 1] * weight for key, value in motion.items() if key != "root_quat_w"
    }
    # Batched shortest-arc SLERP; the native helper accepts a single interpolation pair.
    q0, q1 = quat[i0], quat[i0 + 1]
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    theta = torch.acos(dot.abs().clamp(max=1.0))
    sine = torch.sin(theta)
    w0 = torch.where(sine < 1e-6, 1 - weight, torch.sin((1 - weight) * theta) / (sine + 1e-12))
    w1 = torch.where(sine < 1e-6, weight, torch.sin(weight * theta) / (sine + 1e-12))
    result["root_quat_w"] = F.normalize(w0 * q0 + w1 * q1, dim=-1)
    return result


class MotionLoader:
    """Local or server-hosted reference trajectory sampled at the policy rate.

    NPZ files contain ``fps``, ``joint_names``, ``body_names``, ``joint_pos`` [rad],
    ``joint_vel`` [rad/s], ``body_pos_w`` [m], ``body_quat_w`` (WXYZ),
    ``body_lin_vel_w`` [m/s] and ``body_ang_vel_w`` [rad/s]. Joint arrays have shape
    (frames, joints); body arrays have shape (frames, bodies, 3 or 4). Positions
    use a local world origin; the command adds each environment's origin.

    Legacy files may provide names in a same-stem JSON file with a matching
    ``sha256``. Loaded quaternions use Isaac Lab's XYZW convention.
    """

    def __init__(
        self,
        motion_file: str,
        body_names: Sequence[str],
        joint_names: Sequence[str],
        device: str,
        *,
        forward_kinematics: Callable | None = None,
        source_fps: float = 30.0,
        target_fps: float = 50.0,
        csv: CsvMotionCfg | None = None,
    ):
        path = Path(motion_file)
        if not path.is_dir():
            path = Path(retrieve_file_path(motion_file))
        pose_only = path.is_dir() or path.suffix != ".npz"
        if not pose_only:
            with np.load(path, allow_pickle=False) as data:
                pose_only = "body_pos_w" not in data
        if pose_only:
            if forward_kinematics is None:
                raise ValueError("Root/joint reference inputs require a forward_kinematics callback.")
            motion = MotionLibrary(
                str(path), joint_names, source_fps=source_fps, target_fps=target_fps, csv=csv, device=device
            )
            self.__dict__.update(vars(motion))
            self.fps = target_fps
            state = (
                motion.root_pos_w,
                motion.root_quat_w,
                motion.root_lin_vel_w,
                motion.root_ang_vel_w,
                motion.joint_pos,
                motion.joint_vel,
            )
            with torch.no_grad():
                chunks = [
                    forward_kinematics(tuple(value[i : i + 4096] for value in state))
                    for i in range(0, motion.time_step_total, 4096)
                ]
            for name, values in zip(("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"), zip(*chunks)):
                setattr(self, name, torch.cat(values))
            return
        with np.load(path, allow_pickle=False) as data:
            if "body_names" in data and "joint_names" in data:
                names = {key: data[key].tolist() for key in ("body_names", "joint_names")}
            else:
                metadata_path = retrieve_file_path(os.path.splitext(motion_file)[0] + ".json")
                names = json.loads(Path(metadata_path).read_text())
                if names["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"Motion metadata does not match {path}.")
            for key in ("body_names", "joint_names"):
                if len(names[key]) != len(set(names[key])):
                    raise ValueError(f"Duplicate {key} in {path}.")
            body_indices = [names["body_names"].index(name) for name in body_names]
            joint_indices = [names["joint_names"].index(name) for name in joint_names]
            self.fps = float(data["fps"].item())
            if not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError(f"Invalid motion FPS: {self.fps}.")
            self.time_step_total = len(data["joint_pos"])
            if self.time_step_total < 2:
                raise ValueError("A motion must contain at least two frames.")
            arrays: dict[str, torch.Tensor] = {}
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                values = data[key]
                is_joint = key.startswith("joint_")
                count = len(names["joint_names" if is_joint else "body_names"])
                shape = (self.time_step_total, count)
                if not is_joint:
                    shape += (4 if key == "body_quat_w" else 3,)
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {key} in {path}: expected finite values with shape {shape}.")
                values = values[:, joint_indices if is_joint else body_indices]
                if key == "body_quat_w":
                    if not np.allclose(np.linalg.norm(values, axis=-1), 1.0, atol=1e-3):
                        raise ValueError(f"Non-unit body quaternions in {path}.")
                    values = values[..., [1, 2, 3, 0]]
                arrays[key] = torch.tensor(values, dtype=torch.float32, device=device)
            self.joint_pos: torch.Tensor = arrays["joint_pos"]
            self.joint_vel: torch.Tensor = arrays["joint_vel"]
            self.body_pos_w: torch.Tensor = arrays["body_pos_w"]
            self.body_quat_w: torch.Tensor = arrays["body_quat_w"]
            self.body_lin_vel_w: torch.Tensor = arrays["body_lin_vel_w"]
            self.body_ang_vel_w: torch.Tensor = arrays["body_ang_vel_w"]
            self.motion_starts = torch.zeros(1, dtype=torch.long, device=device)
            self.motion_ends = torch.tensor([self.time_step_total], device=device)
            self.names = [path.stem]
            self.root_pos_w = self.body_pos_w[:, 0]
            self.root_quat_w = self.body_quat_w[:, 0]
            self.root_lin_vel_w = self.body_lin_vel_w[:, 0]
            self.root_ang_vel_w = self.body_ang_vel_w[:, 0]

    def clip_end(self, frames: torch.Tensor) -> torch.Tensor:
        """Exclusive clip end for each frame, without crossing concatenated clip boundaries."""
        return self.motion_ends[torch.searchsorted(self.motion_ends, frames.contiguous(), right=True)]


def _load_pose_motion(
    path: Path, joint_names: Sequence[str], source_fps: float, target_fps: float, csv: CsvMotionCfg
) -> dict[str, torch.Tensor]:
    """Adapt a read-only pose clip to the requested sample rate and native XYZW state layout."""
    dt = 1.0 / target_fps
    if path.suffix == ".csv":
        motion = load_csv_motion(path, joint_names, csv)
    else:
        raw = torch.tensor(np.load(path, allow_pickle=False), dtype=torch.float32)
        width = 7 + len(joint_names)
        if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != width:
            raise ValueError(f"Motion {path} must contain at least two {width}-channel frames.")
        if not torch.isfinite(raw).all() or (raw[:, 3:7].norm(dim=-1) < 1e-6).any():
            raise ValueError(f"Motion {path} contains nonfinite values or invalid quaternions.")
        motion = {"root_pos_w": raw[:, :3], "root_quat_w": raw[:, 3:7], "joint_pos": raw[:, 7:]}
    motion = resample_motion(motion, source_fps, target_fps)
    for velocity, position in (("root_lin_vel_w", "root_pos_w"), ("joint_vel", "joint_pos")):
        if velocity not in motion:
            motion[velocity] = torch.gradient(motion[position], spacing=dt, dim=0)[0]
    R = matrix_from_quat(motion["root_quat_w"])
    if "root_ang_vel_w" not in motion:
        w = torch.zeros_like(motion["root_pos_w"])
        w[1:-1] = axis_angle_from_quat(quat_from_matrix(R[2:] @ R[:-2].transpose(-1, -2))) / (2 * dt)
        w[0] = axis_angle_from_quat(quat_from_matrix(R[1] @ R[0].T)) / dt
        w[-1] = axis_angle_from_quat(quat_from_matrix(R[-1] @ R[-2].T)) / dt
        motion["root_ang_vel_w"] = w
    motion["root_quat_w"] = quat_from_matrix(R)
    return motion


class MotionLibrary:
    """Concatenated clips with exclusive ends, retaining supplied reference velocities.

    The input can be one local file or a directory of clips.
    NPZ clips contain ``joint_pos``, ``joint_vel``, ``root_pos_w``, ``root_quat_w``
    (WXYZ), ``root_lin_vel_w``, and ``root_ang_vel_w``. CSV inputs use
    the configured read-only adapter; NPY inputs use metric XYZW poses. Missing
    velocities are estimated within each clip, never across clip boundaries.
    Stored root quaternions use Isaac Lab's XYZW convention. NPZ files must
    already match ``target_fps`` (50 Hz when their ``fps`` field is omitted).
    Named NPZ joints are reordered to ``joint_names``; unnamed NPZ arrays must
    already use that order.
    """

    def __init__(
        self,
        motion_dir: str,
        joint_names: Sequence[str],
        *,
        source_fps: float = 30.0,
        target_fps: float = 50.0,
        csv: CsvMotionCfg | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
        device: str = "cpu",
    ):
        if not all(np.isfinite(rate) and rate > 0 for rate in (source_fps, target_fps)):
            raise ValueError("Motion frame rates must be finite and positive.")
        names = list(joint_names)
        if len(set(names)) != len(names):
            raise ValueError("Target joint names must be unique.")
        self.device = device
        path = Path(motion_dir)
        directory = path.parent if path.is_file() else path
        files = [path] if path.is_file() else sorted(path.rglob("*.npz"))
        if not files:
            files = sorted((*Path(motion_dir).rglob("*.csv"), *Path(motion_dir).rglob("*.npy")))
        if not files:
            raise ValueError(f"No motion NPZ/CSV/NPY files found in {motion_dir}.")
        clip_names = [str(path.relative_to(directory).with_suffix("")) for path in files]
        if len(set(clip_names)) != len(clip_names):
            raise ValueError("Motion names must be unique across files.")
        if not 0 <= shard_index < shard_count:
            raise ValueError("Motion shard index must be in [0, motion_shard_count).")
        if shard_count > 1:
            files = files[shard_index::shard_count]
            if not files:
                raise ValueError("Sharded references require at least one clip per rank.")
        fields = ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")
        clips = []
        for path in files:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    clip = {key: torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in fields}
                    if "joint_names" in data:
                        source_names = data["joint_names"].tolist()
                        if (
                            data["joint_names"].ndim != 1
                            or len(set(source_names)) != len(source_names)
                            or set(source_names) != set(names)
                        ):
                            raise ValueError(f"Reference joint names must match the target joints in {path}.")
                        indices = [source_names.index(name) for name in names]
                        for key in ("joint_pos", "joint_vel"):
                            if clip[key].ndim != 2 or clip[key].shape[1] != len(source_names):
                                raise ValueError(f"Invalid {key} layout in {path}.")
                            clip[key] = clip[key][:, indices]
                    if (float(data["fps"]) if "fps" in data else 50.0) != target_fps:
                        raise ValueError(f"Reference clip {path} must be sampled at {target_fps} Hz.")
            else:
                clip = {
                    key: value.to(device)
                    for key, value in _load_pose_motion(
                        path, names, source_fps, target_fps, csv or CsvMotionCfg()
                    ).items()
                }
            length = len(clip["joint_pos"])
            widths = (len(names), len(names), 3, 4, 3, 3)
            if length < 2 or any(clip[key].shape != (length, width) for key, width in zip(fields, widths)):
                raise ValueError(f"Invalid reference layout in {path}.")
            if any(not torch.isfinite(value).all() for value in clip.values()):
                raise ValueError(f"Nonfinite reference in {path}.")
            norms = clip["root_quat_w"].norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=0):
                raise ValueError(f"Non-unit root quaternions in {path}.")
            if path.suffix == ".npz":
                clip["root_quat_w"] = clip["root_quat_w"][:, [1, 2, 3, 0]]
            clips.append(clip)
        lengths = [len(clip["joint_pos"]) for clip in clips]
        self.names = [str(path.relative_to(directory).with_suffix("")) for path in files]
        for key in fields:
            setattr(self, key, torch.cat([clip[key] for clip in clips]))
        self.motion_ends = torch.tensor(lengths, device=device).cumsum(0)
        self.motion_starts = self.motion_ends - torch.tensor(lengths, device=device)
        self.time_step_total = sum(lengths)
