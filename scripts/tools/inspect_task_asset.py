# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write a JSON snapshot of an asset's resolved runtime contract inside a task."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(
    description=(
        "Inspect one task asset and write its resolved names, physical properties, and policy interface to JSON."
    )
)
parser.add_argument("--task", required=True, help="Registered task name.")
parser.add_argument("--asset_name", default="robot", help="Scene key of the asset to inspect. Default: robot.")
parser.add_argument("--env_index", type=int, default=0, help="Environment instance to sample. Default: 0.")
parser.add_argument("--output", type=Path, required=True, help="JSON output path.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


_BODY_PROPERTIES = (
    "body_mass",
    "body_inertia",
    "body_com_pose_b",
)
_JOINT_PROPERTIES = (
    "default_joint_pos",
    "joint_stiffness",
    "joint_damping",
    "joint_armature",
    "joint_friction_coeff",
    "joint_pos_limits",
    "joint_vel_limits",
    "joint_effort_limits",
)


def _as_tensor(value: Any) -> torch.Tensor | None:
    """Return a tensor view of a runtime value when one is available."""
    if hasattr(value, "torch"):
        value = value.torch
    return value if isinstance(value, torch.Tensor) else None


def _sample_value(value: Any, env_index: int, num_envs: int) -> Any:
    """Convert a runtime value to JSON data, selecting one environment when batched."""
    tensor = _as_tensor(value)
    if tensor is not None:
        if tensor.ndim > 0 and tensor.shape[0] == num_envs:
            tensor = tensor[env_index]
        return tensor.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _json_safe(value: Any) -> Any:
    """Recursively convert descriptor values to JSON-compatible data."""
    tensor = _as_tensor(value)
    if tensor is not None:
        return tensor.detach().cpu().tolist()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_data_properties(asset: Any, names: tuple[str, ...], env_index: int, num_envs: int) -> dict[str, Any]:
    """Read supported properties from an asset data object."""
    result = {}
    for name in names:
        try:
            value = getattr(asset.data, name)
        except (AttributeError, NotImplementedError):
            continue
        result[name] = _sample_value(value, env_index, num_envs)
    return result


def _physical_diagnostics(asset: Any, env_index: int, num_envs: int) -> dict[str, Any]:
    """Compute basic finite and positivity checks for mass, inertia, and armature."""
    result: dict[str, Any] = {}
    try:
        mass = _as_tensor(asset.data.body_mass)
    except (AttributeError, NotImplementedError):
        mass = None
    if mass is not None:
        mass = mass[env_index] if mass.ndim > 0 and mass.shape[0] == num_envs else mass
        result["body_mass_all_finite"] = bool(torch.isfinite(mass).all())
        result["body_mass_all_positive"] = bool((mass > 0).all())
        result["body_mass_min"] = float(mass.min().item())

    try:
        inertia = _as_tensor(asset.data.body_inertia)
    except (AttributeError, NotImplementedError):
        inertia = None
    if inertia is not None:
        inertia = inertia[env_index] if inertia.ndim > 0 and inertia.shape[0] == num_envs else inertia
        matrices = inertia.reshape(-1, 3, 3)
        symmetric_matrices = 0.5 * (matrices + matrices.transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(symmetric_matrices)
        scale = eigenvalues.abs().amax(dim=-1).clamp_min(torch.finfo(eigenvalues.dtype).tiny)
        tolerance = 10.0 * torch.finfo(eigenvalues.dtype).eps * scale
        triangle_inequalities = eigenvalues[..., 2] <= eigenvalues[..., 0] + eigenvalues[..., 1] + tolerance
        result["body_inertia_all_finite"] = bool(torch.isfinite(inertia).all())
        result["body_inertia_positive_definite"] = bool((eigenvalues > 0).all())
        result["body_inertia_triangle_inequalities"] = bool(triangle_inequalities.all())
        result["body_inertia_min_eigenvalue"] = float(eigenvalues.min().item())

    try:
        body_com_pose_b = _as_tensor(asset.data.body_com_pose_b)
    except (AttributeError, NotImplementedError):
        body_com_pose_b = None
    if body_com_pose_b is not None:
        if body_com_pose_b.ndim > 0 and body_com_pose_b.shape[0] == num_envs:
            body_com_pose_b = body_com_pose_b[env_index]
        result["body_com_pose_b_all_finite"] = bool(torch.isfinite(body_com_pose_b).all())

    try:
        armature = _as_tensor(asset.data.joint_armature)
    except (AttributeError, NotImplementedError):
        armature = None
    if armature is not None:
        armature = armature[env_index] if armature.ndim > 0 and armature.shape[0] == num_envs else armature
        result["joint_armature_all_finite"] = bool(torch.isfinite(armature).all())
        result["joint_armature_all_nonnegative"] = bool((armature >= 0).all())
        result["joint_armature_min"] = float(armature.min().item())
    return result


def _manager_contract(env: Any) -> dict[str, Any]:
    """Read manager term names and dimensions when the task is manager-based."""
    result: dict[str, Any] = {}
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
        result["actions"] = _json_safe(action_manager.get_IO_descriptors)
    observation_manager = getattr(env, "observation_manager", None)
    if observation_manager is not None:
        result["observations"] = _json_safe(observation_manager.get_IO_descriptors)
    return result


def _asset_source(asset: Any) -> str | None:
    """Return the configured source path when the spawner exposes one."""
    spawn_cfg = getattr(getattr(asset, "cfg", None), "spawn", None)
    for name in ("usd_path", "asset_path", "urdf_path", "mjcf_path"):
        value = getattr(spawn_cfg, name, None)
        if value is not None:
            return str(value)
    return None


def main() -> None:
    """Launch the task and write its resolved asset contract."""
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.env_index + 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env.reset()
            unwrapped = env.unwrapped
            if args_cli.asset_name not in unwrapped.scene.keys():
                available = ", ".join(unwrapped.scene.keys())
                raise KeyError(f"Asset '{args_cli.asset_name}' is not in the scene. Available keys: {available}")
            asset = unwrapped.scene[args_cli.asset_name]
            report = {
                "task": args_cli.task,
                "physics_backend": unwrapped.scene.physics_backend,
                "physics_config_type": type(env_cfg.sim.physics).__name__ if env_cfg.sim.physics is not None else None,
                "asset_name": args_cli.asset_name,
                "asset_type": type(asset).__name__,
                "asset_source": _asset_source(asset),
                "env_index": args_cli.env_index,
                "timing": {
                    "physics_dt": env_cfg.sim.dt,
                    "decimation": getattr(env_cfg, "decimation", None),
                    "policy_dt": env_cfg.sim.dt * getattr(env_cfg, "decimation", 1),
                },
                "spaces": {
                    "action": str(env.action_space),
                    "observation": str(env.observation_space),
                },
                "manager_contract": _manager_contract(unwrapped),
                "scene_keys": unwrapped.scene.keys(),
                "body_names": list(getattr(asset, "body_names", [])),
                "joint_names": list(getattr(asset, "joint_names", [])),
                "actuator_names": list(getattr(asset, "actuators", {}).keys()),
                "body_properties": _read_data_properties(
                    asset, _BODY_PROPERTIES, args_cli.env_index, unwrapped.num_envs
                ),
                "joint_properties": _read_data_properties(
                    asset, _JOINT_PROPERTIES, args_cli.env_index, unwrapped.num_envs
                ),
                "diagnostics": _physical_diagnostics(asset, args_cli.env_index, unwrapped.num_envs),
            }
        finally:
            env.close()

    output_path = args_cli.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote asset contract to: {output_path}")


if __name__ == "__main__":
    main()
